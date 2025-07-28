from omegaconf import OmegaConf
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0
from sentence_transformers import SentenceTransformer, util
import torch, json, os
from PIL import Image
import spacy

# Load spaCy English model for NER
nlp = spacy.load("en_core_web_sm")

# Load config
cfg = OmegaConf.load("eval_configs/tst_minigpt4_eval.yaml")
model_cfg = cfg.model

# Build model
model_cls = registry.get_model_class(model_cfg.arch)
model = model_cls.from_config(model_cfg).to("cuda")
model.eval()

# Build visual processor
vis_processor_cfg = cfg.datasets.cc_sbu_align.vis_processor.eval
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor)

# Load eval data
with open("datasetNew/MiniEval/captions.json", "r") as f:
    data = json.load(f)
annotations = data["annotations"]

# Load similarity model
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
SIM_THRESHOLD = 0.75

correct, hallucinated = 0, 0

def extract_entities(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents)

for item in annotations:
    img_name = f"{item['image_id']}.jpg"
    gt_caption = item["caption"]
    img_path = os.path.join("datasetNew/MiniEval/image", img_name)

    if not os.path.exists(img_path):
        print(f"⚠️ Missing: {img_name}")
        continue

    image = Image.open(img_path).convert("RGB")
    conv = CONV_VISION_Vicuna0.copy()
    img_list = []
    chat.upload_img(image, conv, img_list)
    chat.encode_img(img_list)
    response, _ = chat.answer(conv, img_list)

    # Compute semantic similarity
    sim_score = util.cos_sim(
        sim_model.encode(gt_caption, convert_to_tensor=True),
        sim_model.encode(response, convert_to_tensor=True)
    ).item()
    is_correct = sim_score >= SIM_THRESHOLD
    correct += is_correct

    # NER-based hallucination detection
    gt_entities = extract_entities(gt_caption)
    pred_entities = extract_entities(response)
    extra_entities = pred_entities - gt_entities
    is_hallucinated = len(extra_entities) > 0
    hallucinated += is_hallucinated

    print(f"\n🖼 Image: {img_name}")
    print(f"✅ GT: {gt_caption}\n🤖 PR: {response}")
    print(f"🔍 Sim: {sim_score:.2f} | ✅ Correct: {is_correct} | 🔥 Hallucinated: {is_hallucinated}")
    if is_hallucinated:
        print(f"⚠️ Extra Entities: {extra_entities}")

# Final scores
total = len(annotations)
print(f"\n📊 Final Accuracy: {correct}/{total} = {100 * correct / total:.2f}%")
print(f"🔥 Hallucination Rate: {hallucinated}/{total} = {100 * hallucinated / total:.2f}%")
