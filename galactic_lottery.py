import hashlib
import secrets
import base58
import secp256k1
from KeyDatabase import satoshi_keys

# 1. Pre-process target keys into a Python Set of raw bytes
# A Python set uses a hash map, making lookups O(1) (instantaneous).
# Comparing raw bytes is also significantly faster than comparing hex strings.
satoshi_set = {bytes.fromhex(k) for k in satoshi_keys}

def hash_key(public_key_bytes):
    """Hashes the public key using SHA256 then RIPEMD160."""
    sha = hashlib.sha256(public_key_bytes).digest()
    ripe = hashlib.new('ripemd160', sha).digest()
    return ripe

def private_key_to_wif(private_key_bytes, compressed=True):
    """Converts raw private key bytes to Wallet Import Format (WIF)."""
    prefixed_key = b'\x80' + private_key_bytes
    if compressed:
        prefixed_key += b'\x01'
    checksum = hashlib.sha256(hashlib.sha256(prefixed_key).digest()).digest()[:4]
    wif_key = prefixed_key + checksum
    return base58.b58encode(wif_key).decode('utf-8')

def main():
    attempts = 10_000  # We can increase batch size since the CPU is very fast now
    total_attempts = 0
    
    # Max valid space for the SECP256k1 Elliptic Curve
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140

    print(f"[*] Loaded {len(satoshi_set)} target keys into fast-lookup set.")
    print("[*] Starting high-speed CPU search utilizing libsecp256k1...\n")

    while True:
        # Search safely inside the curve bounds
        start_value = secrets.randbelow(N - attempts - 1) + 1
        
        for i in range(attempts):
            current_priv_int = start_value + i
            priv_bytes = current_priv_int.to_bytes(32, byteorder='big')
            
            # 2. Use the highly optimized C-library to do the heavy ECC math
            sk = secp256k1.PrivateKey(priv_bytes)
            vk_bytes = sk.pubkey.serialize(compressed=True)
            
            # 3. Hash the bytes
            ripe_digest = hash_key(vk_bytes)
            
            # 4. Instantaneous O(1) Set Lookup
            if ripe_digest in satoshi_set:
                wif_key = private_key_to_wif(priv_bytes)
                print(f"\n\n[SUCCESS] Winning private key found!")
                print(f"WIF: {wif_key}")
                print(f"Hex: {hex(current_priv_int)}")
                return  # Exit the program
        
        total_attempts += attempts
        # Using comma formatting `:,` to make reading the attempts easier
        print(f"\r{total_attempts:,} attempts made. Current key: {hex(start_value + attempts - 1)}", end='', flush=True)

if __name__ == "__main__":
    main()
