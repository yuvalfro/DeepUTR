import random
import pandas as pd

NUCLEOTIDS = ['A', 'C', 'G', 'T']
TENT = "TTTTTTTTTT"
AUrich = "TATTATTTAT"
PUM = "TATGTAAATATGTA"
PolyU = "TTTTTTTTTTTTT"
miR430 = "GCACTT"
L = 110
N = int((L-10)/2)

initial_value_path = "..DeepUTR/files/dataset/validation_A_minus_normalized_levels.csv"

def randStr(planted_nucs, chars=NUCLEOTIDS):
    if planted_nucs == "TENT":
        mid_seq = TENT
    elif planted_nucs == "AUrich":
        mid_seq = AUrich
    elif planted_nucs == "PUM":
        mid_seq = PUM
    elif planted_nucs == "PolyU":
        mid_seq = PolyU
    elif planted_nucs == "miR430":
        mid_seq = miR430

    n = int((L-len(mid_seq))/2)
    seq = ""
    seq += ''.join(random.choice(chars) for _ in range(n))
    seq += mid_seq
    seq += ''.join(random.choice(chars) for _ in range(n))
    if len(seq) != L:
        seq += ''.join(random.choice(chars) for _ in range(L-len(seq)))
    return seq

def randStrWithA(a, chars=NUCLEOTIDS):
    seq = ""
    seq += ''.join(random.choice(chars) for _ in range(N-a))
    seq += 'A' * a
    seq += TENT
    seq += 'A' * a
    seq += ''.join(random.choice(chars) for _ in range(N-a))
    return seq

df = pd.read_csv(initial_value_path)
avg_initial = round(df['1h'].mean(), 4)
avg_list = []
for i in range(10000):
    avg_list.append(avg_initial)
df_avg = pd.DataFrame(avg_list, columns=['1h'])
df_avg.to_csv("new_seq_initial.csv")

seq_list = []
for i in range(10000):
    seq_list.append(randStr("PolyU"))

# # for i in range(2000):
# #     seq_list.append(randStrWithA(1))
# #     seq_list.append(randStrWithA(2))
# #     seq_list.append(randStrWithA(3))
# #     seq_list.append(randStrWithA(4))
# #     seq_list.append(randStrWithA(5))

df_seq = pd.DataFrame(seq_list, columns=['seq'])
df_seq.to_csv("new_seq_PolyU.csv")


