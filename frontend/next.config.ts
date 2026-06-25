import type { NextConfig } from "next";

const isDev = process.env.NODE_ENV === "development";
const backendUrl = process.env.BACKEND_API_URL || (isDev ? "http://127.0.0.1:5001" : "https://tb-sense.onrender.com");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
      {
        source: "/predict",
        destination: `${backendUrl}/predict`,
      },
      {
        source: "/download-report/:path*",
        destination: `${backendUrl}/download-report/:path*`,
      },
      {
        source: "/dashboard/static/uploads/:path*",
        destination: `${backendUrl}/dashboard/static/uploads/:path*`,
      },
    ];
  },
};

export default nextConfig;
