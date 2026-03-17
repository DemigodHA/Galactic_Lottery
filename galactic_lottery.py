import hashlib
import secrets
import base58
import numpy as np
import pyopencl as cl
from ecdsa import SigningKey, SECP256k1
from KeyDatabase import satoshi_keys

# 1. Pre-process the target keys into a 2D numpy array for OpenCL compatibility
# A 160-bit RIPEMD-160 hash fits exactly into five 32-bit unsigned integers.
satoshi_array = np.zeros((len(satoshi_keys), 5), dtype=np.uint32)
for i, hex_str in enumerate(satoshi_keys):
    for j in range(5):
        satoshi_array[i, j] = int(hex_str[j*8:(j+1)*8], 16)

# 2. OpenCL C Kernel Code (Runs directly on the Intel Iris Xe Execution Units)
kernel_code = """
__kernel void check_key_gpu(__global const unsigned int *hashed_keys,
                            __global const unsigned int *target_keys,
                            __global int *found_key,
                            const int num_hashed,
                            const int num_targets) {
    // Get the global ID (the current thread/key index being processed)
    int idx = get_global_id(0);
    
    // Protect against out-of-bounds memory access
    if (idx < num_hashed) {
        for (int i = 0; i < num_targets; i++) {
            int match = 1;
            // Compare the 5 chunks of 32-bit integers
            for (int j = 0; j < 5; j++) {
                if (hashed_keys[idx * 5 + j] != target_keys[i * 5 + j]) {
                    match = 0;
                    break;
                }
            }
            
            if (match == 1) {
                // If found, write the index back to the result buffer
                found_key[0] = idx;
                return; // Stop checking
            }
        }
    }
}
"""

def get_intel_gpu_context():
    """Automatically searches the system for an Intel GPU to bind to."""
    try:
        for platform in cl.get_platforms():
            for device in platform.get_devices(cl.device_type.GPU):
                if "Intel" in device.vendor or "Intel" in device.name:
                    print(f"[*] Successfully bound to GPU: {device.name}")
                    return cl.Context([device])
    except Exception:
        pass
    
    print("[!] Auto-select failed. Prompting for manual device selection...")
    return cl.create_some_context()

def private_key_to_wif(private_key, compressed=True):
    private_key_bytes = private_key.to_bytes(32, byteorder='big')
    prefixed_key = b'\x80' + private_key_bytes
    if compressed:
        prefixed_key += b'\x01'
    checksum = hashlib.sha256(hashlib.sha256(prefixed_key).digest()).digest()[:4]
    wif_key = prefixed_key + checksum
    return base58.b58encode(wif_key).decode('utf-8')

def main():
    attempts = 1_000
    total_attempts = 0
    
    # Max valid space for SECP256k1 Curve
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140
    
    # Initialize OpenCL context and execution queue
    ctx = get_intel_gpu_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    # Compile the OpenCL C program for your Intel Hardware Just-In-Time
    prg = cl.Program(ctx, kernel_code).build()
    
    # Extract the kernel ONCE to avoid repeated retrieval overhead
    check_kernel = cl.Kernel(prg, "check_key_gpu")
    
    # Send the target database to Intel GPU memory (Only needs to happen once)
    d_satoshi_keys = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=satoshi_array)

    while True:
        # Search safely inside the curve bounds
        start_value = secrets.randbelow(N - attempts - 1) + 1
        private_keys = [start_value + i for i in range(attempts)]
        
        hashed_keys_array = np.zeros((attempts, 5), dtype=np.uint32)
        
        # Elliptic Curve Math & Hashing on the CPU
        for i, private_key in enumerate(private_keys):
            sk = SigningKey.from_secret_exponent(private_key, curve=SECP256k1)
            vk_bytes = sk.verifying_key.to_string("compressed")
            
            sha = hashlib.sha256(vk_bytes).digest()
            ripe = hashlib.new('ripemd160', sha).digest()
            
            for j in range(5):
                hashed_keys_array[i, j] = int.from_bytes(ripe[j*4:(j+1)*4], byteorder='big')
        
        # Setup inputs and output buffers for the GPU block
        d_hashed_keys = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=hashed_keys_array)
        
        found_arr = np.array([-1], dtype=np.int32)
        d_found_key = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=found_arr)
        
        # Fire the GPU command payload
        check_kernel(
            queue, 
            (attempts,),      # Global work size
            None,             # Auto-calculate local block size
            d_hashed_keys, 
            d_satoshi_keys, 
            d_found_key, 
            np.int32(attempts), 
            np.int32(len(satoshi_keys))
        )
        
        # Pull the integer result back into Python memory over PCIe
        cl.enqueue_copy(queue, found_arr, d_found_key)
        queue.finish()  # Synchronize
        
        found_idx = found_arr[0]
        
        if found_idx != -1:
            winning_private_key = private_keys[found_idx]
            wif_key = private_key_to_wif(winning_private_key)
            print(f"\n\n[SUCCESS] Winning private key found!\nWIF: {wif_key}")
            print(f"Hex: {hex(winning_private_key)}")
            break
        
        total_attempts += attempts
        print(f"\r{total_attempts} attempts made so far. Current key: {hex(start_value + attempts - 1)}", end='', flush=True)

if __name__ == "__main__":
    main()
