from sentence_transformers import SentenceTransformer
import os

# تحديد المسار الذي تريد حفظ الموديل فيه
model_save_path = "./models"

# تنزيل الموديل من Hugging Face
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# حفظ الموديل محلياً
model.save(model_save_path)
print(f"تم حفظ الموديل بنجاح في: {model_save_path}")