"use client"
import { useState } from 'react';

export default function Home() {
  const [video, setVideo] = useState<File | null>(null);
  const [uploadedVideo, setUploadedVideo] = useState<string | null>(null);

  const handleUpload = async () => {
    if (!video) return;

    const formData = new FormData();
    formData.append('video', video);

    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setUploadedVideo(data.filePath);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center p-6">
      <h1 className="text-3xl font-bold text-gray-700 mb-4">Video Uploader</h1>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => setVideo(e.target.files?.[0] || null)}
        className="mb-4 p-2 border rounded"
      />
      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
      >
        Upload Video
      </button>
      {uploadedVideo && (
      <div className="mt-6">
      <h2 className="text-xl font-semibold">Uploaded Video:</h2>
      <video controls className="mt-4 w-full max-w-md rounded shadow-lg">
        <source src={uploadedVideo} type="video/mp4" />
      </video>
    </div>
  )}
    </div>
  );
}
