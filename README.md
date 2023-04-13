
## Step 1 - create file lists
```bash
aws  s3 ls  "s3://s-laion/papers/medarxiv/"  --recursive | grep ".meca"|awk '{print "s3://s-laion/"$4}'> medarxiv_file_list.txt
aws  s3 ls  "s3://s-laion/papers/biorxiv/"  --recursive | grep ".meca"|awk '{print "s3://s-laion/"$4}'> biorxiv_file_list.txt
```

## Step 2: alllocate

```bash
salloc --partition cpu16 --cpus-per-task 16
srun --cpus-per-task 16 --comment laion --pty /bin/bash -i 
```

## Step 3: run

```bash
time python main.py extract-figure-caption-pairs medarxiv_file_list.txt --nb-shards=2500 --path-shards=medarxiv_figure_captions --num-workers=16

time python main.py extract-figure-caption-pairs bioarxiv_file_list.txt --nb-shards=2500 --path-shards=biorxiv_figure_captions --num-workers=16
```

## Step 4: copy

```bash
aws s3 sync medarxiv_figure_captions s3://s-laion/papers/
aws s3 sync biorxiv_figure_captions s3://s-laion/papers/
```