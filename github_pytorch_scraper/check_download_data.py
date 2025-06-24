import json

with open('/matx/u/simonguo/gh_triton_scrape/metadata.json', 'r') as f:
    try:
        data = json.load(f)
        print(f"JSON is well-formed")

        print("================================================")
        print("\nProject names:")
        for filename, project in data.items():
            print(f"- {project['full_name']}")
        print("================================================")

        print(f"Number of entries: {len(data)}")

    except json.JSONDecodeError as e:
        print(f"JSON is malformed: {e}")