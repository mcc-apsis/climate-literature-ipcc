from climate_literature.constants import RAW_DATA, MAP_DATA
from climate_literature.settings import settings

rule fetch_scopus_query:
    output:
        directory(RAW_DATA),
    shell:
        f'rsync -av --progress -e "ssh -o ProxyJump={settings.ts01_username}@ts01.pik-potsdam.de" {settings.ts01_username}@se164:/data/academic-api/data/results/8/responses/*.jsonl {output}/'

rule fetch_map_data:
    output:
        directory(MAP_DATA)
    shell:
        f"rsync -av --progress galm@10.10.12.41:/mnt/bulk/project-data/climate-policy-map-data/data/ {output}/"