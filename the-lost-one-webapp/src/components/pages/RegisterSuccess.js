// RegisterSuccess.js
import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const RegisterSuccess = () => {
  const navigate = useNavigate();

  useEffect(() => {
    setTimeout(() => {
      navigate("/");
    }, 3000);
  }, [navigate]);

  return (
    <div>
      <h1>Registered Successfully!</h1>
    </div>
  );
};

export default RegisterSuccess;
