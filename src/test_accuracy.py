import json

if __name__ == "__main__":
    inp = open("test_all.txt", "r")
    n_q = 0
    n_acc = 0
    for st in inp:
        if st.strip() == "":
            break
        n_q += 1
        line = st.strip().split("\t")
        q = line[0]
        ans = line[1]
        line = q.strip().split("_")
        path = line[0]
        num = int(line[1])
        obj = json.load(open(path, "r"))
        if obj["answers"][num] == ans:
            n_acc += 1
        #else:
        print q, obj["answers"][num], ans
    print n_acc * 1. / n_q