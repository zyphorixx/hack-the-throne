import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { GeistMono } from "geist/font/mono"
import { Suspense } from "react"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
})

export const metadata: Metadata = {
  title: "ForgetMetNot",
  description: "AI-powered webcam streaming application"
}

import ConvexClientProvider from "@/components/ConvexClientProvider";

// ... (imports remain the same, just adding the provider import if not already there, actually I'll insert it at the top level separately if needed, but here I replace the body)

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`font-sans ${inter.variable} ${GeistMono.variable}`}>
        <ConvexClientProvider>
          <Suspense fallback={<div>Loading...</div>}>{children}</Suspense>
        </ConvexClientProvider>
      </body>
    </html>
  )
}
