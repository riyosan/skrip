import streamlit as st


class Toc:

    def __init__(self):
        self._items = []
        self._placeholder = None
    
    def title(self, text):
        self._markdown(text, "h1")

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = "".join(filter(str.isalnum, text)).lower()

        st.markdown(f"<{level} id='{key}'>{text}</{level}>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()

st.title("Table of contents")
toc.placeholder()

toc.title("Title")

for a in range(10):
    st.write("Blabla...")

toc.header("Header 1")

for a in range(10):
    st.write("Blabla...")

toc.header("Header 2")

for a in range(10):
    st.write("Blabla...")

toc.subheader("Subheader 1")

for a in range(10):
    st.write("Blabla...")

toc.subheader("Subheader 2")

for a in range(10):
    st.write("Blabla...")

toc.generate()









# import streamlit as st
# from functools import wraps


# def main():
#     pages = {
#         "Home": page_home,
#         "Settings": page_settings,
#     }

#     # If 'page' is not present, setup default values for settings widgets.
#     if "page" not in st.session_state:
#         st.session_state.update({
#             # Default page
#             "page": "Home",

#             # Radio, selectbox and multiselect options
#             "options": ["Hello", "Everyone", "Happy", "Streamlit-ing"],

#             # Default widget values
#             "text": "",
#             "slider": 0,
#             "checkbox": False,
#             "radio": "Hello",
#             "selectbox": "Hello",
#             "multiselect": ["Hello", "Everyone"],
#         })


#     with st.sidebar:
#         page = st.radio("Select your page", tuple(pages.keys()))

#     pages[page]()


# def page_home():
#     st.write(f"""
#     # Settings values
#     - **Input**: {st.session_state.text}
#     - **Slider**: `{st.session_state.slider}`
#     - **Checkbox**: `{st.session_state.checkbox}`
#     - **Radio**: {st.session_state.radio}
#     - **Selectbox**: {st.session_state.selectbox}
#     - **Multiselect**: {", ".join(st.session_state.multiselect)}
#     """)


# def page_settings():
#     st.title("Change settings")

#     st.text_input("Input", key="text")
#     st.slider("Slider", 0, 10, key="slider")
#     st.checkbox("Checkbox", key="checkbox")
#     st.radio("Radio", st.session_state["options"], key="radio")
#     st.selectbox("Selectbox", st.session_state["options"], key="selectbox")
#     st.multiselect("Multiselect", st.session_state["options"], key="multiselect")


# def track_forbidden_keys(widget):
#     if "__track_forbidden_keys__" not in widget.__dict__:
#         widget.__dict__["__track_forbidden_keys__"] = True

#         @wraps(widget)
#         def wrapper_widget(*args, **kwargs):
#             if "key" in kwargs:
#                 st.session_state._forbidden_keys.add(kwargs["key"])
#             return widget(*args, **kwargs)

#         return wrapper_widget

#     return widget


# def legacy_session_state():
#     """Restore Streamlit v0.88.0 session state behavior.
#     The legacy session state behavior allowed widget states to be persistent,
#     even after their disappearance.
#     """
#     # Initialize forbidden keys set.
#     if "_forbidden_keys" not in st.session_state:
#         st.session_state._forbidden_keys = set()

#     # Self-assign session state items that are not in our forbidden set.
#     # This actually translates widget state items to user-defined session
#     # state items internally, which makes widget states persistent.
#     for key, value in st.session_state.items():
#         if key not in st.session_state._forbidden_keys:
#             st.session_state[key] = value

#     # We don't want to self-assign keys from the following widgets
#     # to avoid a Streamlit API exception.
#     # So we wrap them and save any used key in our _forbidden_keys set.
#     st.button = track_forbidden_keys(st.button)
#     st.download_button = track_forbidden_keys(st.download_button)
#     st.file_uploader = track_forbidden_keys(st.file_uploader)
#     st.form = track_forbidden_keys(st.form)

#     # We can clear our set to avoid keeping unused widget keys over time.
#     st.session_state._forbidden_keys.clear()


# if __name__ == "__main__":
#     legacy_session_state()
#     main()