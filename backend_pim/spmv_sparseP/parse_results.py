import os
import sys
import argparse



def save_csv(results: dict, keys, file_path, repeat_cnt):
    chart = open(file_path, "w")
    chart.write(", ".join(["Repeat"] + keys) + "\n")
    avg_data = [0] * len(keys)
    for rep in range(repeat_cnt):
        current_data = [str(rep)]
        for i, key in enumerate(keys):
            val = results.get(key)[rep]
            current_data.append(val)
            avg_data[i] += float(val) / repeat_cnt
        assert(len(current_data) == len(keys) + 1)
        chart.write(", ".join(current_data) + "\n")
    chart.write(", ".join(["avg"] + [str(item) for item in avg_data]) + "\n")
    chart.close()
    return avg_data



def get_data(out_dir, csv_dir):
    os.makedirs(csv_dir, exist_ok=True)
    #for subdir, dirs, files in os.walk(rootdir+"/"+resultdir):
    avs_results = {}
    for file in os.listdir(out_dir):
        if (not file.endswith("out")):
            continue
        print(file)
        out_name = file[:-4] + ".csv"
        out_path = os.path.join(csv_dir, out_name)

        results = {}
        repeat_cnt = 0
        keys = []
        with open(os.path.join(out_dir, file), "r") as ins:
            for line in ins:
                if (line.startswith("-------------------- Repeat")):
                    repeat_cnt += 1
                if not line.startswith("[DATA]"):
                    continue
                label, value = line[6:-1].split(": ")
                if not (label in results):
                    results[label] = []
                    keys.append(label)
                results[label].append(value)

        total_time_key = "pim_time_spmm(ms)"
        load_sparse_key = "load_sparse_time"
        new_key = "pim_time_dense(ms)"
        new_val = []

        keys.append(new_key)
        for i in range(repeat_cnt):
            new_val.append(str(float(results[total_time_key][i]) - float(results[load_sparse_key][i])))
        results[new_key] = new_val

        avg = save_csv(results, keys, out_path, repeat_cnt)
        avs_results[file[:-4]] = avg
    if len(avs_results.items()[0][0].split("_")[1:]) == 6:
        all_labels = ["dtype", "balance", "dataset", "hidden_size", "sp_parts", "ds_parts"] + keys
    elif len(avs_results.items()[0][0].split("_")[1:]) == 9:
        all_labels = ["sp_format", "dtype", "balance", "balance_tsklt", "nr_tasklets", "dataset", "hidden_size", "sp_parts", "ds_parts"] + keys
    chart = open(os.path.join(csv_dir, "average_all.csv"), "w")
    chart.write(", ".join(all_labels) + "\n")
    for item in avs_results.items():
        part1 = item[0].split("_")[1:]
        part2 = [str(it) for it in item[1]]
        chart.write(", ".join(part1 + part2) + "\n")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, default="./results_230905_070021")
    parser.add_argument('--csv_path', type=str, default='')
    args = parser.parse_args()
    if not os.path.isdir(args.out_path):
        print("[ERROR] {} is not a directory!".format(args.out_path))
        exit(1)
    if args.csv_path == "":
        args.csv_path = os.path.join(args.out_path, "csv_result")
    print("Processing .out file in {}".format(args.out_path))
    print("Csv result will be stored in {}".format(args.csv_path))
    get_data(args.out_path, args.csv_path)

if __name__ == "__main__":
    main()
