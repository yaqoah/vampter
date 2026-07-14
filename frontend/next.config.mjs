/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {
    root: "frontend",
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
