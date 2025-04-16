// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDF-NVn8vPex-eDtBSQLwUUnNKnaCc6wIE",
  authDomain: "the-lost-one-auth.firebaseapp.com",
  projectId: "the-lost-one-auth",
  storageBucket: "the-lost-one-auth.appspot.com",
  messagingSenderId: "331206502071",
  appId: "1:331206502071:web:4909bd85e5661c49b2cf32",
  measurementId: "G-K2KMLQK6N0",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

const analytics = getAnalytics(app);

export { auth };
