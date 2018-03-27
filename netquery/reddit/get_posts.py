from Queue import Empty
from multiprocessing import Queue, Process


def worker(pid, queue, in_format, out_format):

    while True:
        try:
            index = queue.get(block=False)
        except Empty:
            break
        print "Proc", pid, "starting", index
        with open(in_format.format(index=index)) as in_fp:
            with open(out_format.format(index=index), "w") as out_fp:
                in_fp.readline()
                for line in in_fp:
                    line_info = line.split(",")
                    date_info = line_info[0].split()[0].split("-")
                    if date_info[0] != "2016" or date_info[1] != "05":
                        continue
                    day = int(date_info[2])
                    if day <= 5:
                        out_fp.write(line)
        print "Proc", pid, "finished", index

def run_parallel(num_procs, in_format, out_format, index_range):
    queue = Queue()
    for index in index_range:
        queue.put(index)
    procs = [Process(target=worker, args=[i,queue,in_format,out_format]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

if __name__ == "__main__":
    run_parallel(30, "/dfs/scratch0/dataset/20180122-Reddit/data/stanford_submission_data/stanford_submission_data{index:012d}.csv",
            "/dfs/scratch0/nqe-reddit/post_data/{index:03d}.csv",
            range(139))
