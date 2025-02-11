## Project Popcorn Synthetic Data Generation

This is repo synthetically generate pytorch programs and then create <pytorch, torch inductor> pairs to form a dataset

### Requirements

```
uv pip install -r requirements.txt
```

### Running the code

```bash
# clean out the inductor cache
rm -rf /tmp/torchinductor_<username>/
rm generated/*

export OPENAI_API_KEY=<openai api key>

python generate_code_random_torch.py --num_files <some number we used 3000>
python clean.py --input_dir generated
python create_dataset.py --gen_dir_path generated --uuid_file filtered_uuids.json
# you should now have a dataset in the format of dataset.json
```

### Todo:
- [ ] Put clean.py into generate_code.py
