import json

json_path = '../data/github_repos/neuromorphic_repos.json'

with open(json_path, 'r') as f:
    repos_data = json.load(f)

repos_data = repos_data['repositories']

for repo_type in ['pytorch', 'hdl']:
    print(f'mkdir -p ../data/github_repos/{repo_type}')
    for repo in repos_data[repo_type]:
        full_name = repo['full_name']
        clone_url = repo['url'] + '.git'
        print(f'git clone {clone_url} ../data/github_repos/{repo_type}/{full_name.replace("/", "_")}')
        