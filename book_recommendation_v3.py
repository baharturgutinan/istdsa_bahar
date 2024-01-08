import pickle
import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config (
    layout="wide",
    page_title="Book Recommender System",
    page_icon="https://www.realsimple.com/thmb/KrGb42aamhHKaMzWt1Om7U42QsY=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/great-books-for-anytime-2000-4ff4221eb1e54b659689fef7d5e265d5.jpg",
    menu_items={
        
        "About": "For More Information\n" + "https://github.com/baharturgutinan/istdsa_bahar"
    }
)

st.header('Book Recommender System  :100:')


st.markdown("""A book store wants to increase their sales by recommending books to the readers according to their current choose. Let's help them with our model.
            
Welcome and Thank you for using our recommendation model.
            
Below (:point_down:) you can find two buttons, which of the first makes recommendations based on people who has liked the same books with you before.
            
The second button will help you to discover the books that has similar attributes with the book you have selected from the box.

Have an enjoyable reading!! :sparkles:

""")

books=pd.read_csv('C:\\Users\\bahar.inan\\Documents\\ISTDSA_Bahar_Projeler\\proje5\\BX-Books.csv',sep=';',on_bad_lines='skip', encoding='latin-1',low_memory=False)
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher',
       'Image-URL-L']]


books.rename(columns= {
    "Book-Title":"title",
    "Book-Author":"author",
    "Year-Of-Publication":"year",
    "Publisher":"publisher",
    "Image-URL-L":"img_url"} , inplace=True)

model = pickle.load(open('cosine_sim.pkl','rb'))
book_names = pickle.load(open('books_name.pkl','rb'))
final_rating = pickle.load(open('final_rating.pkl','rb'))
book_pivot = pickle.load(open('book_pivot.pkl','rb'))
cosine_sim_df = pickle.load(open('cosine_sim_df.pkl','rb'))
#books = pickle.load(open('books.pkl','rb'))
cosine_sim, df = joblib.load('content_based_recommender_model.pkl')

books_url=final_rating.loc[:,['title','img_url']]

def find_similar_books(book, count=1):
       
    books_summed = cosine_sim_df[book]
    books_summed = books_summed.sort_values(ascending=False) # Yüksek skorlar daha iyi
    ranked_books = books_summed.index[books_summed.index.isin(books)==False]
    ranked_books = ranked_books.tolist()

    if count is None:
        return ranked_books
    else:
        return ranked_books[:count]


def get_recommendations(book):
    idx = df.index[df['title'] == book].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[book_indices].to_list()



selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

suggested_books_user=find_similar_books(selected_books, 5)
suggested_books=get_recommendations(selected_books)

book_url_user=[]
for book in suggested_books_user:
    book_url_user.append(books_url[books_url.title==book]['img_url'].tolist()[1])


book_url=[]
for book in suggested_books:
    book_url.append(books_url[books_url.title==book]['img_url'].tolist()[1])


if st.button('Show Recommendation Based on Similar Users :red_haired_woman: :older_man: :man: :red_haired_man:'): 	
  
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text(suggested_books_user[1])
        st.image(book_url_user[1])
    with col2:
        st.text(suggested_books_user[2])
        st.image(book_url_user[2])
    with col3:
        st.text(suggested_books_user[3])
        st.image(book_url_user[3])


if st.button('Show Recommendation Based on Similar Books :green_book: :blue_book: :orange_book: :closed_book:'):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text(suggested_books[1])
        st.image(book_url[1])
        
    with col2:
        st.text(suggested_books[2])
        st.image(book_url[2])
        
    with col3:
        st.text(suggested_books[3])
        st.image(book_url[3])
        

st.sidebar.image("https://images.unsplash.com/photo-1568667256549-094345857637?q=80&w=1000&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8bGlicmFyeXxlbnwwfHwwfHx8MA%3D%3D")




