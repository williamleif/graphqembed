
with open("videogame_subreddits.txt", "r") as fp:
    with open("videogame_subreddits_clean.txt", "w") as out_fp:
        for line in fp:
            if "/r/" in line:
                out_fp.write(line.strip().split("/")[-1] + "\n")
