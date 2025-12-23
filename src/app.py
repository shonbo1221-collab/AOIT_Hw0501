"""
AI Text Detection - Streamlit Web Application

A web interface for detecting whether text is AI-generated or human-written.
"""

import streamlit as st
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.predict import AITextDetector


# Page configuration
st.set_page_config(
    page_title="AI Text Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .ai-result {
        background-color: #ffe6e6;
        border-left: 5px solid #ff4444;
    }
    .human-result {
        background-color: #e6f7ff;
        border-left: 5px solid #4444ff;
    }
    .metric-container {
        text-align: center;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Load the AI text detector model (cached)."""
    try:
        return AITextDetector()
    except FileNotFoundError as e:
        st.error(f"❌ Model not found: {str(e)}")
        st.info("Please run `python src/model/train.py` first to train the model.")
        st.stop()


def get_confidence_color(confidence):
    """Get color based on confidence level."""
    if confidence >= 80:
        return "#00cc00"  # Green
    elif confidence >= 60:
        return "#ffaa00"  # Orange
    else:
        return "#ff4444"  # Red


def main():
    """Main application function."""
    
    # Header
    st.markdown('<div class="main-header">🤖 AI Text Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze text to determine if it\'s AI-generated or human-written</div>', unsafe_allow_html=True)
    
    # Load model
    detector = load_detector()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This system uses machine learning to classify text as either:
        - 🤖 **AI-generated**
        - 👤 **Human-written**
        
        **How it works:**
        - TF-IDF feature extraction
        - Logistic Regression classifier
        - Trained on sample datasets
        """)
        
        st.header("📊 Model Info")
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Features:** TF-IDF (5000 features)")
        st.write("**N-grams:** Unigrams + Bigrams")
        
        st.header("💡 Tips")
        st.write("""
        - Enter at least 5 words for accurate results
        - Longer texts generally produce better predictions
        - Confidence scores indicate prediction certainty
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Enter Text to Analyze")
        
        # Text input area
        user_text = st.text_area(
            "Paste or type your text here:",
            height=250,
            placeholder="Enter text to analyze (minimum 5 words)...",
            help="Enter the text you want to classify"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
        
        # Sample texts section
        st.subheader("📚 Try Sample Texts")
        
        col_sample1, col_sample2 = st.columns(2)
        
        with col_sample1:
            if st.button("Sample AI Text", use_container_width=True):
                st.session_state.sample_text = """Artificial intelligence has revolutionized numerous industries in recent years. The technology enables machines to perform tasks that traditionally required human intelligence. Machine learning algorithms process vast amounts of data to identify patterns and make predictions. These systems continuously improve their performance through experience and exposure to new information."""
        
        with col_sample2:
            if st.button("Sample Human Text", use_container_width=True):
                st.session_state.sample_text = """So I was walking to the coffee shop this morning, right? And this guy literally bumps into me while staring at his phone. Doesn't even say sorry! Like, come on dude, at least look up once in a while. Anyway, got my usual latte and it was actually pretty good today."""
        
        # Display sample text if selected
        if 'sample_text' in st.session_state:
            st.info("Sample text loaded! Click 'Analyze Text' to see results.")
            user_text = st.session_state.sample_text
            st.text_area("Selected sample:", value=user_text, height=150, disabled=True)
    
    with col2:
        st.header("📊 Analysis Results")
        
        # Placeholder for results
        results_placeholder = st.empty()
        
        # Perform analysis when button is clicked
        if analyze_button or ('sample_text' in st.session_state and user_text):
            if not user_text or len(user_text.strip()) == 0:
                st.error("❌ Please enter some text to analyze.")
            elif len(user_text.split()) < 5:
                st.warning("⚠️ Text is too short. Please enter at least 5 words for accurate analysis.")
            else:
                with st.spinner("🔄 Analyzing text..."):
                    # Get prediction
                    result = detector.predict_text(user_text)
                    
                    # Check for errors
                    if 'error' in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Display results
                        label = result['label']
                        confidence = result['confidence']
                        ai_prob = result['ai_probability']
                        human_prob = result['human_probability']
                        
                        # Result box styling
                        box_class = "ai-result" if label == "AI-generated" else "human-result"
                        icon = "🤖" if label == "AI-generated" else "👤"
                        
                        st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
                        
                        # Main prediction
                        st.markdown(f"### {icon} {label}")
                        st.markdown(f"**Confidence:** {confidence:.1f}%")
                        
                        # Confidence meter
                        st.progress(confidence / 100)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Detailed probabilities
                        st.subheader("🎯 Probability Breakdown")
                        
                        col_ai, col_human = st.columns(2)
                        
                        with col_ai:
                            st.metric(
                                label="🤖 AI-generated",
                                value=f"{ai_prob*100:.1f}%",
                                delta=None
                            )
                        
                        with col_human:
                            st.metric(
                                label="👤 Human-written",
                                value=f"{human_prob*100:.1f}%",
                                delta=None
                            )
                        
                        # Confidence interpretation
                        st.subheader("📈 Confidence Level")
                        if confidence >= 80:
                            st.success("✅ High confidence - Very reliable prediction")
                        elif confidence >= 60:
                            st.warning("⚠️ Medium confidence - Moderately reliable")
                        else:
                            st.error("❌ Low confidence - Less reliable prediction")
                        
                        # Text statistics
                        st.subheader("📊 Text Statistics")
                        word_count = len(user_text.split())
                        char_count = len(user_text)
                        st.write(f"**Words:** {word_count}")
                        st.write(f"**Characters:** {char_count}")
                
                # Clear sample text from session state
                if 'sample_text' in st.session_state:
                    del st.session_state.sample_text
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>AI Text Detection System | Built with Streamlit & scikit-learn</p>
            <p><small>Note: This is a demonstration system. Results may vary based on training data.</small></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
