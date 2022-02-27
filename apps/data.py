import streamlit as st
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from apps import praproses
import os
from dateutil import parser
import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
def app():
  if 'main_data.csv' not in os.listdir('data'):
    st.markdown("Please upload data through `Home` page!")
  else:
    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameter'):
      split_size = st.sidebar.slider('Rasio Pembagian Data (% Untuk Data Latih)', 10, 90, 80, 5)
      jumlah_fitur = st.sidebar.slider('jumlah pilihan fitur (Untuk Data Latih)', 5, 47, 20, 5)
      parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 10, 100, 50, 10)
      k = st.sidebar.slider('Jumlah K (KNN)', 11, 101, 55, 11)
      
    if st.sidebar.button('Latih & Uji'):  
      df = pd.read_csv('data/main_data.csv')
      df1=pd.read_csv('data/df1.csv')
      from sklearn.feature_selection import mutual_info_classif
      #determine the mutual information
      mutual_info = mutual_info_classif(df.drop(columns=['enrolled']), df.enrolled)
      mutual_info = pd.Series(mutual_info)
      mutual_info.index = df.drop(columns=['enrolled']).columns
      mutual_info.sort_values(ascending=False)
      from sklearn.feature_selection import SelectKBest
      fitur_terpilih = SelectKBest(mutual_info_classif, k=jumlah_fitur)
      fitur_terpilih.fit(df.drop(columns=['enrolled']), df.enrolled)
      pilhan_kolom = df.drop(columns=['enrolled']).columns[fitur_terpilih.get_support()]
      pd.Series(pilhan_kolom).to_csv('data/fitur_pilihan.csv',index=False)
      fitur = pilhan_kolom.tolist()
      baru = df[fitur]
      from sklearn.preprocessing import StandardScaler
      sc_X = StandardScaler()
      pilhan_kolom = sc_X.fit_transform(baru)

      import joblib
      joblib.dump(sc_X, 'data/minmax_scaler.joblib')

      X_train, X_test, y_train, y_test = train_test_split(pilhan_kolom, df['enrolled'],test_size=(100-split_size)/100, random_state=111)
      
      # st.write(X_test)

      from sklearn.naive_bayes import GaussianNB
      nb = GaussianNB() # Define classifier)
      nb.fit(X_train, y_train)

      # Make predictions
      y_test_pred = nb.predict(X_test)

      # Evaluate model
      nb_container = st.columns((1.1, 0.9))
      matrik = (classification_report(y_test, y_test_pred))

      #page layout
      with nb_container[0]:
        st.write("2a. Naive Bayes report using sklearn")
        st.text('Naive Bayes Report:\n ' + matrik)
      st.write(" ")
      st.write(" ")
      st.write(" ")

      with nb_container[1]:
        cm_label = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
        cm_label.index.name = 'Actual'
        cm_label.columns.name = 'Predicted'
        sns.heatmap(cm_label, annot=True, cmap='Blues', fmt='g')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
      st.write(" ")
      st.write(" ")

      rf = RandomForestClassifier(n_estimators=parameter_n_estimators, max_depth=2, random_state=42) # Define classifier
      rf.fit(X_train, y_train) # Train model

      # Make predictions
      y_test_pred = rf.predict(X_test)

      # Evaluate model
      nb_container = st.columns((1.1, 0.9))
      matrik = (classification_report(y_test, y_test_pred))

      #page layout
      with nb_container[0]:
        st.write("2b. Random Forest report using sklearn")
        st.text('Random Forest Report:\n ' + matrik)
      st.write(" ")
      st.write(" ")
      st.write(" ")

      with nb_container[1]:
        cm_label = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
        cm_label.index.name = 'Actual'
        cm_label.columns.name = 'Predicted'
        sns.heatmap(cm_label, annot=True, cmap='Blues', fmt='g')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
      st.write(" ")
      st.write(" ")


      estimator_list = [
          ('nb',nb),
          ('rf',rf)]

      # Build stack model
      stack_model = StackingClassifier(
          estimators=estimator_list, final_estimator=KNeighborsClassifier(k),cv=5
      )

      # Train stacked model
      stack_model.fit(X_train, y_train)

      # Make predictions
      y_test_pred = stack_model.predict(X_test)

      # Evaluate model
      nb_container = st.columns((1.1, 0.9))
      matrik = (classification_report(y_test, y_test_pred))

      #page layout
      with nb_container[0]:
        st.write("2c. Stack report using sklearn")
        st.text('Stack Report:\n ' + matrik)
      st.write(" ")
      st.write(" ")
      st.write(" ")

      with nb_container[1]:
        cm_label = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=np.unique(y_test), index=np.unique(y_test))
        cm_label.index.name = 'Actual'
        cm_label.columns.name = 'Predicted'
        sns.heatmap(cm_label, annot=True, cmap='Blues', fmt='g')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
      st.write(" ")
      st.write(" ")      
      
      joblib.dump(stack_model, 'data/stack_model.pkl')

      var_enrolled = df1['enrolled']
      #membagi menjadi train dan test untuk mencari user id
      X_train, X_test, y_train, y_test = train_test_split(df1, df1['enrolled'], test_size=(100-split_size)/100, random_state=111)
      train_id = X_train['user']
      test_id = X_test['user']
      #menggabungkan semua
      y_pred_series = pd.Series(y_test).rename('Aktual',inplace=True)
      hasil_akhir = pd.concat([y_pred_series, test_id], axis=1).dropna()
      hasil_akhir['Prediksi']=y_test_pred
      hasil_akhir = hasil_akhir[['user','Aktual','Prediksi']].reset_index(drop=True)
      st.text('Tabel Perbandingan Asli dan Prediksi:\n ')
      st.write(hasil_akhir)
