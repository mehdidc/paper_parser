
# Medarxiv

## Step 1 - Download raw data

`aws  s3 sync s3://biorxiv-src-monthly s3://s-laion/papers/biorxiv --request-payer`

## Step 2 - create file list

```bash
aws  s3 ls  "s3://s-laion/papers/medarxiv/"  --recursive | grep ".meca"|awk '{print "s3://s-laion/"$4}'> medarxiv_file_list.txt
```

## Step 3: run

```bash
salloc --partition cpu16 --cpus-per-task 16
srun --cpus-per-task 16 --comment laion --pty /bin/bash -i 
time python main.py extract medarxiv_file_list.txt --nb-shards=2500 --path-shards=medarxiv_figure_captions --num-workers=16 --processor=medarxiv
```
## Step 4: copy

```bash
aws s3 sync medarxiv_figure_captions s3://s-laion/papers/medarxiv_figure_captions
```

Options to make to make it quicker: <https://github.com/aws/aws-cli/blob/develop/awscli/topics/s3-config.rst>


# Biorxiv

## Step 1 - Download raw data

`aws  s3 sync s3://medrxiv-src-monthly s3://s-laion/papers/medrxiv --request-payer`

## Step 2 - create file list

`aws  s3 ls  "s3://s-laion/papers/biorxiv/"  --recursive | grep ".meca"|awk '{print "s3://s-laion/"$4}'> biorxiv_file_list.txt`

## Step 3 - run

```bash
salloc --partition cpu16 --cpus-per-task 16
srun --cpus-per-task 16 --comment laion --pty /bin/bash -i 
time python main.py extract biorxiv_file_list.txt --nb-shards=2500 --path-shards=biorxiv_figure_captions --num-workers=16 --processor=biorxiv
```
## Step 4 - copy

`aws s3 sync biorxiv_figure_captions s3://s-laion/papers/biorxiv_figure_captions`


# PubMed


## Step 1 - Create file list


```bash
wget https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt
cat oa_file_list.txt|awk '{print "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/"$1}' > pubmed_file_list.txt
```

## Step 2 - run

```bash
salloc --partition cpu16 --cpus-per-task 16
srun --cpus-per-task 16 --comment laion --pty /bin/bash -i 
time python main.py extract pubmed_file_list.txt --nb-shards=2500 --path-shards=pubmed_figure_captions --num-workers=16
```