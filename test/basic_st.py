import streamlit as st

# basics
st.title("Welcome brother!")
st.write("This is just the intro LOL")

# user input
name = st.text_input("Your name?")
if name:
    st.write(f"Welcome, {name}! let's play")
    
age= st.slider("Your age", 0,100)
gender= st.radio("Gender",("Male", "Female", "Others"))

interest= st.selectbox("Interested in ", ("CV","ML" ,"Both"))

if st.button("Create Profile"):
    st.write(f"Age: {age}, Gender: {gender}, Interest: {interest}")


# problem---

# all_profile = []
# new_profile={"Name": name,
#               "Age": age,
#               "Gender": gender,
#               "Interest": interest}

# all_profile.append(new_profile)

# if st.button("Show all Profile"):
#     st.write(all_profile)


#solution

if "profiles" not in st.session_state:
    st.session_state.profiles= []

new_profile={"Name": name,
              "Age": age,
              "Gender": gender,
              "Interest": interest}

if st.button("Save and show profile"):
    st.session_state.profiles.append(new_profile)

st.write("All profiles:", st.session_state.profiles)