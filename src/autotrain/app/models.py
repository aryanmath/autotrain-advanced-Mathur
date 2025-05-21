import collections
import logging

from huggingface_hub import list_models, HfApi


def get_sorted_models(hub_models):
    """
    Filters and sorts a list of models based on their download count.

    Args:
        hub_models (list): A list of model objects. Each model object must have the attributes 'id', 'downloads', and 'private'.

    Returns:
        list: A list of model IDs sorted by their download count in descending order. Only includes models that are not private.
    """
    hub_models = [{"id": m.id, "downloads": m.downloads} for m in hub_models if m.private is False]
    hub_models = sorted(hub_models, key=lambda x: x["downloads"], reverse=True)
    hub_models = [m["id"] for m in hub_models]
    return hub_models


def _fetch_text_classification_models():
    """
    Fetches and sorts text classification models from the Hugging Face model hub.

    This function retrieves models for the tasks "fill-mask" and "text-classification"
    from the Hugging Face model hub, sorts them by the number of downloads, and combines
    them into a single list. Additionally, it fetches trending models based on the number
    of likes in the past 7 days, sorts them, and places them at the beginning of the list
    if they are not already included.

    Returns:
        list: A sorted list of model identifiers from the Hugging Face model hub.
    """
    hub_models1 = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models2 = list(
        list_models(
            task="text-classification",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_llm_models():
    hub_models = list(
        list_models(
            task="text-generation",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = get_sorted_models(hub_models)
    trending_models = list(
        list_models(
            task="text-generation",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_image_classification_models():
    hub_models = list(
        list_models(
            task="image-classification",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="image-classification",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_image_object_detection_models():
    hub_models = list(
        list_models(
            task="object-detection",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
            pipeline_tag="object-detection",
        )
    )
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="object-detection",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
            pipeline_tag="object-detection",
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_seq2seq_models():
    hub_models = list(
        list_models(
            task="text2text-generation",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = get_sorted_models(hub_models)
    trending_models = list(
        list_models(
            task="text2text-generation",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_token_classification_models():
    hub_models1 = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models2 = list(
        list_models(
            task="token-classification",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
        )
    )
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models

    return hub_models


def _fetch_st_models():
    hub_models1 = list(
        list_models(
            task="sentence-similarity",
            library="sentence-transformers",
            sort="downloads",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    hub_models2 = list(
        list_models(
            task="fill-mask",
            library="transformers",
            sort="downloads",
            direction=-1,
            limit=30,
            full=False,
        )
    )

    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models = list(
        list_models(
            task="sentence-similarity",
            library="sentence-transformers",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
        )
    )
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_vlm_models():
    hub_models1 = list(
        list_models(
            task="image-text-to-text",
            sort="downloads",
            direction=-1,
            limit=100,
            full=False,
            filter=["paligemma"],
        )
    )
    # hub_models2 = list(
    #     list_models(
    #         task="image-text-to-text",
    #         sort="downloads",
    #         direction=-1,
    #         limit=100,
    #         full=False,
    #         filter=["florence2"],
    #     )
    # )
    hub_models2 = []
    hub_models = list(hub_models1) + list(hub_models2)
    hub_models = get_sorted_models(hub_models)

    trending_models1 = list(
        list_models(
            task="image-text-to-text",
            sort="likes7d",
            direction=-1,
            limit=30,
            full=False,
            filter=["paligemma"],
        )
    )
    # trending_models2 = list(
    #     list_models(
    #         task="image-text-to-text",
    #         sort="likes7d",
    #         direction=-1,
    #         limit=30,
    #         full=False,
    #         filter=["florence2"],
    #     )
    # )
    trending_models2 = []
    trending_models = list(trending_models1) + list(trending_models2)
    if len(trending_models) > 0:
        trending_models = get_sorted_models(trending_models)
        hub_models = [m for m in hub_models if m not in trending_models]
        hub_models = trending_models + hub_models
    return hub_models


def _fetch_asr_models():
    """Fetch ASR models from Hugging Face Hub."""
    models = []
    try:
        api = HfApi()
        # Get models sorted by downloads
        asr_models = api.list_models(
            filter="automatic-speech-recognition",
            sort="downloads",
            direction=-1,
            limit=50
        )
        models = [model.modelId for model in asr_models]
        
        # Get trending models
        trending_models = api.list_models(
            filter="automatic-speech-recognition",
            sort="likes",
            direction=-1,
            limit=10
        )
        trending = [model.modelId for model in trending_models]
        
        # Add trending models at the beginning if not already included
        for model in trending:
            if model not in models:
                models.insert(0, model)
                
    except Exception as e:
        logging.error(f"Error fetching ASR models: {e}")
        # Fallback to default models
        models = [
            "facebook/wav2vec2-base-960h",
            "facebook/wav2vec2-large-960h",
            "facebook/wav2vec2-large-xlsr-53",
            "microsoft/wavlm-base"
        ]
    
    return models


def fetch_models():
    """Fetch models for all tasks."""
    MODEL_CHOICE = {}
    MODEL_CHOICE["text-classification"] = _fetch_text_classification_models()
    MODEL_CHOICE["token-classification"] = _fetch_token_classification_models()
    MODEL_CHOICE["text-regression"] = _fetch_text_regression_models()
    MODEL_CHOICE["seq2seq"] = _fetch_seq2seq_models()
    MODEL_CHOICE["image-classification"] = _fetch_image_classification_models()
    MODEL_CHOICE["image-regression"] = _fetch_image_regression_models()
    MODEL_CHOICE["image-object-detection"] = _fetch_object_detection_models()
    MODEL_CHOICE["automatic-speech-recognition"] = _fetch_asr_models()
    return MODEL_CHOICE
