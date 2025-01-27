## Project Popcorn Synthetic Data Generation

### Requirements

```
uv pip install -r requirements.txt
```

### Running the code

```bash
# clean out the inductor cache
rm -rf /tmp/torchinductor_<username>/
rm generated/*
rm 
python generate_code.py --num_files <some number we used 3000>
python clean.py --input_dir generated
python create_dataset.py --gen_dir_path generated --uuid_file filtered_uuids.json
# you should now have a dataset in the format of dataset.json
```

### Todo:
- [ ] Put clean.py into generate_code.py
