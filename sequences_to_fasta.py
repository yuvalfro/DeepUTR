seq_PATH = "../DeepUTR-main/files/dataset/validation_seq.csv"
file = open(seq_PATH)
Lines = file.readlines()
fasta_file = open("../DeepUTR/seq.fa", "w")
before = "GGATGCTAGGAGATCTGAGTTCAAGGAT"
after = "ATCTAGAACTATAGTGAGTCGTATTACA"
count = 0
# Strips the newline character
for line in Lines:
    if count == 0:
        count += 1
        continue
    seq = line.split(',')[1][:-1]
    fasta_file.write(f">{count}")
    fasta_file.write("\n")
    fasta_file.write(before)
    fasta_file.write(seq)
    fasta_file.write(after)
    fasta_file.write("\n")
    count += 1
