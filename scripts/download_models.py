from transformers import pipeline

from climate_literature.settings import settings

for model in settings.models:
    print(f"Downloading {model}...")
    try:
        pipeline(
            "text-classification", model=model
        )  # Downloads model + tokenizer + config
        print(f"✓ {model}")
    except Exception as e:
        print(f"✗ {model}: {e}")
