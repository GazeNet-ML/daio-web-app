import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};

const config: NextConfig = {
  serverActions: {
    bodySizeLimit: '200mb', // Adjust size limit here
  },
};

export default config;
