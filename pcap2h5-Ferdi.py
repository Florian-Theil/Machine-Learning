import gzip
import dpkt
import struct

import json
import pandas as pd
import numpy as np

import h5py


SECURITY_EMBEDDINGS = "embeddings.json"
HEADER_SIZE = 40
DATAFRAME_SIZE = 500_000
TIMEDELTA = pd.Timedelta(minutes=1)


def get_udp_payload_from_packet(packet_bytes: bytes):
    """
    Parse a raw Ethernet frame and return the UDP payload (bytes)
    Returns None if not an Ethernet/IP/UDP packet.
    """
    try:
        eth = dpkt.ethernet.Ethernet(packet_bytes)
    except Exception:
        return None

    # Handle IPv4
    ip = eth.data
    # dpkt may represent IPv6 differently
    if isinstance(ip, dpkt.ip.IP):
        transport = ip.data
    elif isinstance(ip, dpkt.ip6.IP6):
        transport = ip.data
    else:
        return None

    # We only care about UDP here
    if not isinstance(transport, dpkt.udp.UDP):
        return None

    udp = transport
    # dpkt.udp.UDP object has .data (payload)
    return bytes(udp.data)

def unpack_quote_update_message(udp_payload: bytes):
    unpacked_data = struct.unpack('<BBQ8sIQQI', udp_payload)
    return unpacked_data

def read_pcap(pcap_path):
    """
    Read pcapng file which is compressed using gzip

    Yield Trade Report MessageInteger
    """
    with gzip.open(pcap_path, "rb") as f:
        for _, buf in dpkt.pcapng.Reader(f):
            udp_payload = get_udp_payload_from_packet(buf)
            if udp_payload is None:  # Was invalid UDP packet
                continue

            if len(udp_payload) < HEADER_SIZE + 3:  # Empty UDP Packet
                continue

            # Find message length and extract message
            udp_payload_len, = struct.unpack_from("<H", udp_payload, 40)
            udp_payload = udp_payload[HEADER_SIZE+2: HEADER_SIZE+udp_payload_len+2]

            # Find message type
            mt, = struct.unpack_from("<B", udp_payload, 0)

            # if mt == 0x54:
            #     yield unpack_trade_report_message(udp_payload)

            if mt == 0x51:
                yield unpack_quote_update_message(udp_payload)


def write_h5(pcap_path):
    data_reader = read_pcap(pcap_path)

    # Read the first message to determine the data shape/type
    first_msg = next(data_reader)
    first_msg = np.asarray(first_msg)

    # Create a resizable dataset
    with h5py.File('my_hdf5_file.h5', 'w') as f:
        dset = f.create_dataset(
            "iex",
            data=np.expand_dims(first_msg, axis=0),
            maxshape=(None,) + first_msg.shape,
            compression="gzip"
        )

        buffer = []
        batch_size = DATAFRAME_SIZE
        for message in data_reader:
            buffer.append(message)
            if len(buffer) >= batch_size:
                new_data = np.asarray(buffer)
                old_size = dset.shape[0]
                dset.resize(old_size + new_data.shape[0], axis=0)
                dset[old_size:] = new_data
                buffer.clear()



def pcap_chunk(pcap_path):
    data_reader = read_pcap(pcap_path)
    headers = ["Message Type", "Sale Condition Flags", "Timestamp", "Symbol", "Bid Size", "Bid Price", "Ask Price", "Ask Size"]

    while True:
        data = [next(data_reader)]
        end_time = data[0][2] + TIMEDELTA.to_timedelta64()

        for report in enumerate(data_reader):
            data.append(report)

            if report[2] >= end_time:
                break

        # process
        yield df



def process():
    df = pd.DataFrame(data, columns=headers)

    df.drop(columns=["Message Type"], inplace=True)
    df["Symbol"] = df["Symbol"].map(hash)
    df['Timestamp'] = pd.to_datetime(df["Timestamp"].astype(int), unit='ns')
    df = df.set_index('Timestamp')

    data["midprice"] = (data["Bid Price"] + data["Ask Price"]) / 2
    data["spread"] = data["Ask Price"] - data["Bid Price"]
    data["imbalance"] = (
        (data["Bid Size"] - data["Ask Size"]) /
        (data["Bid Size"] + data["Ask Size"]).replace(0, pd.NA)
    )
    data.dropna(inplace=True)

    data["midprice_diff"] = (data.groupby("Symbol")["midprice"].diff())
    data["spread_ma"] = (data.groupby("Symbol")["spread"].transform(lambda x: x.rolling(5).mean()))
    data["imbalance_ma"] = (data.groupby("Symbol")["imbalance"].transform(lambda x: x.rolling(5).mean()))
    data.dropna(inplace=True)
