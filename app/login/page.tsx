"use client"
import React from 'react'
import { useState } from 'react';

const login = () => {
  const [isLogin, setIsLogin] = useState(true);

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="w-full max-w-md p-6 bg-white rounded-lg shadow-md">
        <h2 className="text-3xl font-semibold text-center mb-4 text-primary">
          {isLogin ? "Login" : "Sign Up"}
        </h2>
        <form className="space-y-4">
          {!isLogin && (
            <div className='p-1'>
              <label className="block pb-2 pl-1 text-md font-medium text-black">Name</label>
              <input
                className="w-full p-2 border rounded-2xl text-gray-600"
                placeholder="Name"
              />
            </div>
          )}
          <div className='p-1'>
            <label className="block pb-2 pl-1 text-md font-medium text-black">Email</label>
            <input
              type="email"
              className="w-full p-2 border rounded-2xl text-gray-600"
              placeholder="Email"
            />
          </div>
          <div className='p-1'>
            <label className="block pb-2 pl-1 text-md font-medium text-black">Password</label>
            <input
              type="password"
              className="w-full p-2 border rounded-2xl text-gray-600"
              placeholder="Password"
            />
          </div>
          <button type="button" className="w-full bg-primary text-white p-2 rounded-2xl hover:bg-blue-600">
            {isLogin ? "Login" : "Sign Up"}
          </button>
        </form>
        <p className="text-sm text-center mt-4 text-black">
          {isLogin ? "Don't have an account?" : "Already have an account?"} {" "}
          <button onClick={() => setIsLogin(!isLogin)} className="text-primary underline">
            {isLogin ? "Sign Up." : "Login."}
          </button>
        </p>
      </div>
    </div>
  );
}

export default login;
