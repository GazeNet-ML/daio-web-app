"use client"

import { signIn, useSession } from "next-auth/react"
import React from "react"
import UploadVideo from "./UploadVideo" // Import the UploadVideo component

const Dashboard = () => {
    const { data: session } = useSession();

    return (
        <div className="container mx-auto p-4">
            {session ? (
                <>
                    <h1 className="text-2xl font-bold mb-6">Welcome, {session.user?.name}</h1>
                    <UploadVideo />
                </>
            ) : (
                <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
                    <h1 className="text-2xl font-bold">You're not logged in</h1>
                    <div className="flex space-x-4">
                        <button 
                            onClick={() => signIn("google")} 
                            className="border border-black px-4 py-2 rounded-lg hover:bg-blue-800"
                        >
                            Sign in with Google
                        </button>
                        <button 
                            onClick={() => signIn("github")} 
                            className="border border-black px-4 py-2 rounded-lg hover:bg-blue-800"
                        >
                            Sign in with GitHub
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default Dashboard