import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load pre-trained model and vectorizer
def load_model_and_vectorizer():
    try:
        with open("vectorizer.pkl", "rb") as vec_file:
            vectorizer = pickle.load(vec_file)

        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
    except FileNotFoundError:
        st.error("Required files (vectorizer.pkl or model.pkl) are missing.")
        st.stop()

    if not hasattr(vectorizer, "vocabulary_"):
        st.error("The vectorizer is not properly fitted. Check your training process.")
        st.stop()

    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Define label mapping
label_mapping = {
    0: "Fraud",
    1: "Logistic",
    2: "OTP",
    3: "Power Bill",
    4: "Recharge",
    5: "Spam",
    6: "Transactional",
}

# App title
st.title("Message Classifier App")

# Input message
message = st.text_input("Enter a message to classify")

# Classifying the message and suggesting actions
if st.button("Classify Message"):
    if message.strip():
        try:
            # Transform the input message and predict
            predicted_label = model.predict(vectorizer.transform([message]))[0]
            category = label_mapping.get(predicted_label, "Unknown")
            st.write(f"**Message Category:** {category}")

            # Suggest actions based on message category
            if category == "Transactional":
                if "bill" in message.lower():
                    if st.button("Pay Now"):
                        st.write("Redirecting to payment portal...")
                    st.write("Suggested Action: Offer discounts for frequent bill payers.")
                elif "recharge" in message.lower():
                    if st.button("Recharge Now"):
                        st.write("Redirecting to recharge portal...")
                    st.write("Suggested Action: Offer recharge promotions for frequent shoppers.")
                elif "book" in message.lower():
                    if st.button("Book Tickets"):
                        st.write("Redirecting to booking portal...")
                    st.write("Suggested Action: Offer discounts for frequent ticket bookers.")
                else:
                    st.info("No specific action available for this transactional message.")
                    st.write("Suggested Action: Offer recurring payment discounts.")
            elif category == "Logistic":
                if st.button("Track Shipment"):
                    st.write("Redirecting to shipment tracking...")
                st.write("Suggested Action: Provide shipment tracking services for logistic-related inquiries.")
            elif category == "OTP":
                expiry_time = st.slider("Set OTP Expiry Time (minutes)", 1, 30, 5)
                st.write(f"OTP will auto-delete after {expiry_time} minutes.")
                st.write("Suggested Action: Offer instant OTP generation and expiration options.")
            elif category == "Power Bill":
                if st.button("Pay Now"):
                    st.write("Redirecting to payment portal...")
                st.write("Suggested Action: Offer incentives for timely bill payments and discounts for loyal customers.")
            elif category == "Recharge":
                if st.button("Recharge Now"):
                    st.write("Redirecting to recharge portal...")
                st.write("Suggested Action: Offer recharge discounts or promotional bundles for frequent users.")
            elif category == "Fraud":
                st.write("Fraud detected. Improving fraud detection and prevention.")
                st.write("Suggested Action: Improve fraud detection and implement more secure transactions.")
            elif category == "Spam":
                st.write("Message classified as Spam. Consider implementing stricter spam filters.")
                st.write("Suggested Action: Strengthen spam filtering system.")
        except Exception as e:
            st.error(f"Error during classification: {e}")
    else:
        st.warning("Please enter a valid message.")
