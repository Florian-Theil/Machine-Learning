import struct
import binascii
import pandas as pd
import os

# --- Paths ---
data_directory = r"C:\Users\Florian Theil\OneDrive - University of Warwick\Historical data\IEX"
file_path = os.path.join(data_directory, "payloads.txt")

# --- Storage for decoded messages ---
records = []

# --- Open the hex payload file ---
with open(file_path, encoding="ascii", errors="ignore") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        # Must be valid hex and even-length
        if len(line) % 2 != 0 or any(c not in "0123456789abcdefABCDEF" for c in line):
            continue
        try:
            payload = binascii.unhexlify(line)
        except binascii.Error:
            continue  # skip malformed line

        # --- Each payload starts with 2-byte message count ---
        if len(payload) < 2:
            continue
        msg_count = struct.unpack(">H", payload[0:2])[0]
        i = 2  # start after header

        # --- Parse that many messages ---
        for _ in range(msg_count):
            if i >= len(payload):
                break
            mtype = payload[i:i+1].decode(errors="ignore")
            i += 1

            # --- Quote Update ('Q') ---
            if mtype == "Q":
                msg_len = 39 - 1  # 38 more bytes after type
                if i + msg_len > len(payload):
                    break
                msg = payload[i-1:i+msg_len]
                stock   = msg[1:9].decode('ascii', errors="ignore").strip()
                bid_px  = struct.unpack(">I", msg[9:13])[0] / 10000
                bid_sz  = struct.unpack(">I", msg[13:17])[0]
                ask_px  = struct.unpack(">I", msg[17:21])[0] / 10000
                ask_sz  = struct.unpack(">I", msg[21:25])[0]
                records.append({
                    "type": "Q",
                    "symbol": stock,
                    "bid_px": bid_px,
                    "bid_sz": bid_sz,
                    "ask_px": ask_px,
                    "ask_sz": ask_sz,
                })
                i += msg_len
                continue

            # --- Trade Report ('T') ---
            elif mtype == "T":
                msg_len = 44 - 1  # 43 more bytes after type
                if i + msg_len > len(payload):
                    break
                msg = payload[i-1:i+msg_len]
                stock = msg[1:9].decode('ascii', errors="ignore").strip()
                price = struct.unpack(">I", msg[9:13])[0] / 10000
                size  = struct.unpack(">I", msg[13:17])[0]
                records.append({
                    "type": "T",
                    "symbol": stock,
                    "price": price,
                    "size": size,
                })
                i += msg_len
                continue

            # --- Unknown message type ---
            else:
                # Skip unknown messages safely:
                # For real work, you can print or log them
                # print(f"Skipping unknown type {mtype!r} at offset {i}")
                break

# --- Save all parsed messages ---
df = pd.DataFrame(records)
output_file = os.path.join(data_directory, "iex_messages.csv")
df.to_csv(output_file, index=False)

print(f"Wrote {len(df)} rows to {output_file}")