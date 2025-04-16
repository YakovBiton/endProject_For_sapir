import React from "react";
import { Link } from "react-router-dom";
import "../../Styles/Home.css";

function Home() {
  return (
    <body className="home">
      <div className="headerContainer">
        <h2>We Are Here For You!</h2>
        <h1>
          We provide a platform for parents seeking assistance in finding their
          lost children.
        </h1>
        <h1>
          Together, we can make a positive impact and bring loved ones back
          home.
        </h1>
        <div className="btn-side">
          <Link to="/register">
            <button> Sign Up </button>
          </Link>
        </div>
      </div>
    </body>
  );
}

export default Home;
