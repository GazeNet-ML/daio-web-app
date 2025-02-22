"use client";
import React, { useState } from "react";

const login = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState(""); // Only for signup
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Function to handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const endpoint = isLogin ? "/api/auth/login" : "/api/auth/register";
      const body = isLogin ? { email, password } : { email, password, name };
      console.log("Calling API:", endpoint);
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await response.json();
      setLoading(false);

      if (!response.ok) {
        setError(data.error || "Something went wrong.");
        return;
      }

      if (isLogin) {
        localStorage.setItem("token", data.token);
        alert("Login successful!");
      } else {
        alert("Registration successful! Please log in.");
        setIsLogin(true);
      }
    } catch (err) {
      setLoading(false);
      setError("Network error. Please try again.");
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-md">
        <h2 className="text-3xl font-semibold text-center mb-4 text-primary">
          {isLogin ? "Login" : "Sign Up"}
        </h2>

        {error && <p className="text-red-500 text-center">{error}</p>}

        <form className="space-y-4" onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="p-1">
              <label className="block pb-2 pl-1 text-md font-medium text-black">Name</label>
              <input
                className="w-full p-2 border rounded-2xl text-gray-600"
                placeholder="Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required={!isLogin}
              />
            </div>
          )}
          <div className="p-1">
            <label className="block pb-2 pl-1 text-md font-medium text-black">Email</label>
            <input
              type="email"
              className="w-full p-2 border rounded-2xl text-gray-600"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="p-1">
            <label className="block pb-2 pl-1 text-md font-medium text-black">Password</label>
            <input
              type="password"
              className="w-full p-2 border rounded-2xl text-gray-600"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button
            type="submit"
            className="w-full bg-primary text-white p-2 rounded-2xl hover:bg-blue-600"
            disabled={loading}
          >
            {loading ? "Processing..." : isLogin ? "Login" : "Sign Up"}
          </button>
        </form>

        <p className="text-sm text-center mt-4 text-black">
          {isLogin ? "Don't have an account?" : "Already have an account?"}{" "}
          <button onClick={() => setIsLogin(!isLogin)} className="text-primary underline">
            {isLogin ? "Sign Up." : "Login."}
          </button>
        </p>
      </div>
    </div>
  );
};

export default login;


