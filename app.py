import streamlit as st
from model_handler import fashion_system
from utils import log_interaction, get_random_image
import re
from PIL import Image
import time
import pandas as pd

# Page Config
st.set_page_config(page_title="AI 穿搭顧問 (Reflection Agent)", page_icon="🧥", layout="wide")

# --- Helper Functions ---

@st.cache_resource
def load_models():
    """
    Load models once and cache the resource.
    """
    print("[System] 開始載入模型...")
    log_interaction("System", "Loading models...")
    fashion_system.load_models()
    
    if fashion_system.llm_pipeline is None:
        print("[System] LLM 載入失敗！")
        log_interaction("System", "LLM Load Failed.")
        # We don't raise here to allow the app to run with just classifier, 
        # but the user will see the error in logs.
    else:
        log_interaction("System", "Models loaded.")
        print("[System] 模型載入完成。")
    return True

def extract_score(text):
    """
    Attempts to extract a score (0-10) from the text.
    """
    match = re.search(r"(\d+)(?:/10)?", text)
    if match:
        return int(match.group(1))
    return 5 # Default if parsing fails

# --- Main App ---

st.title("🧥 AI 穿搭顧問 (Reflection Agent)")
st.markdown("上傳一張照片或選擇隨機範例。AI 代理將會進行分類、提供穿搭建議、自我反思批評，並給出修正後的最終建議。")

# Sidebar for controls
with st.sidebar:
    st.header("控制面板")
    if st.button("🎲 隨機範例"):
        img_path = get_random_image()
        if img_path:
            st.session_state['selected_image'] = img_path
            st.session_state['uploaded_file'] = None # Clear upload if random is picked
            log_interaction("User", f"Selected random image: {img_path}")
            print(f"[User] 選擇了隨機圖片: {img_path}")
    
    uploaded_file = st.file_uploader("或上傳圖片", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['selected_image'] = None # Clear random if upload is present
        print("[User] 上傳了新圖片")

    analyze_btn = st.button("🚀 開始分析與推薦", type="primary")

# Determine which image to show/process
image_to_process = None
display_image = None

if st.session_state.get('uploaded_file'):
    image_to_process = st.session_state['uploaded_file'] # Streamlit UploadedFile object
    display_image = Image.open(st.session_state['uploaded_file'])
elif st.session_state.get('selected_image'):
    image_to_process = st.session_state['selected_image'] # Path string
    display_image = Image.open(st.session_state['selected_image'])

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("輸入圖片")
    if display_image:
        st.image(display_image, width="stretch")
    else:
        st.info("請上傳圖片或選擇隨機範例。")

with col2:
    st.subheader("分析結果")
    
    if analyze_btn and image_to_process:
        # Use a status container for better progress visibility
        status_container = st.status("正在進行 AI 分析...", expanded=True)
        
        try:
            # 0. Load Models
            status_container.write("⚙️ 正在檢查與載入模型...")
            load_models()

            # 1. Classify
            status_container.write("🔍 正在識別衣物種類...")
            print("[Step 1] 開始分類圖片...")
            
            if isinstance(image_to_process, str):
                pil_image = Image.open(image_to_process)
            else:
                pil_image = Image.open(image_to_process)

            log_interaction("User", "Uploaded image for classification.")
            
            # Get all results (top_k=None returns all)
            classification_results = fashion_system.classify_image(pil_image, top_k=None)
            
            # Get top label for the flow
            if isinstance(classification_results, list) and len(classification_results) > 0:
                top_result = classification_results[0]
                label = top_result['label']
                score = top_result['score']
            else:
                label = "Unknown"
                score = 0.0
                classification_results = []

            log_interaction("Model", f"Classified as: {label}")
            print(f"[Step 1] 分類結果: {label}")
            
            st.success(f"**識別結果:** {label} (信心分數: {score:.1%})")
            
            # Display probabilities
            with st.expander("📊 查看詳細分類機率 (Classification Probabilities)", expanded=True):
                if classification_results:
                    # Create DataFrame for display
                    df_probs = pd.DataFrame(classification_results)
                    # Rename columns for better display
                    df_probs.columns = ["類別", "信心分數"]
                    # Sort by score just in case
                    df_probs = df_probs.sort_values(by="信心分數", ascending=False)
                    
                    # Display as a dataframe
                    st.dataframe(
                        df_probs.style.format({"信心分數": "{:.2%}"}), 
                        width="stretch"
                    )
                    
                    # Display as a bar chart
                    st.bar_chart(df_probs.set_index("類別"))
            
            # 2. Draft Recommendation
            status_container.write("📝 正在生成初步建議 (Draft)...")
            print("[Step 2] 生成初步建議...")
            draft_prompt = f"你是一位時尚造型師。使用者穿著 {label}。請建議適合的搭配風格。請用繁體中文回答。"
            draft_rec = fashion_system.generate_text(draft_prompt)
            log_interaction("Model (Draft)", draft_rec)
            print(f"[Step 2] 初步建議完成 (長度: {len(draft_rec)})")
            
            with st.expander("初步建議 (Draft Recommendation)", expanded=False):
                st.write(draft_rec)
            
            # 3. Reflection (Critique & Score)
            status_container.write("🤔 正在進行自我反思與批評 (Critique)...")
            print("[Step 3] 進行反思批評...")
            critique_prompt = f"扮演一位嚴格的時尚評論家。請對以下建議進行評分 1 到 10 分 (格式: Score: X/10) 並提供簡短的批評。建議內容: {draft_rec}。請用繁體中文回答。"
            critique = fashion_system.generate_text(critique_prompt)
            initial_score = extract_score(critique)
            log_interaction("Model (Critic)", f"Score: {initial_score}, Critique: {critique}")
            print(f"[Step 3] 反思完成 (分數: {initial_score})")
            
            with st.expander("反思評論 (Critique)", expanded=True):
                st.info(critique)
            
            # 4. Refine
            status_container.write("✨ 正在根據反思優化建議 (Refine)...")
            print("[Step 4] 優化建議中...")
            refine_prompt = f"你是一位熱心的造型師。請根據批評改進以下建議。原始建議: {draft_rec}。批評: {critique}。請提供最終潤飾後的建議。請用繁體中文回答。"
            final_rec = fashion_system.generate_text(refine_prompt)
            log_interaction("Model (Final)", final_rec)
            print(f"[Step 4] 最終建議完成")
            
            st.markdown("### 🌟 最終穿搭建議")
            st.write(final_rec)
            
            # 5. Final Score (Self-Evaluation)
            status_container.write("📊 計算最終評分...")
            print("[Step 5] 計算最終分數...")
            final_eval_prompt = f"請對這個最終建議進行評分 1 到 10 分 (格式: Score: X/10)。建議內容: {final_rec}"
            final_eval = fashion_system.generate_text(final_eval_prompt)
            final_score = extract_score(final_eval)
            print(f"[Step 5] 最終分數: {final_score}")
            
            # Regression Metric (Improvement)
            improvement = final_score - initial_score
            
            status_container.update(label="✅ 分析完成！", state="complete", expanded=False)
            
            # Metrics Display
            m1, m2 = st.columns(2)
            m1.metric("初始評分 (Initial Score)", f"{initial_score}/10")
            m2.metric("最終評分 (Final Score)", f"{final_score}/10", delta=improvement)
            
        except Exception as e:
            status_container.update(label="❌ 發生錯誤", state="error")
            st.error(f"分析過程中發生錯誤: {str(e)}")
            print(f"[Error] {str(e)}")
            log_interaction("System", f"Error: {str(e)}")
