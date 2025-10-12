import binascii, struct

line = "01000480010000000000934f0000000000000000000000000100000000000000d55c3dde721c6d18"
payload = binascii.unhexlify(line)

print("Length:", len(payload))
print("Hex bytes:", payload[:32])
print("Integers:", list(payload[:16]))
