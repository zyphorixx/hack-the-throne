import WebcamStream from "@/components/webcam-stream"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { LayoutDashboard } from "lucide-react"

export default function Home() {
  return (
    <main className="h-screen w-screen overflow-hidden relative">
      <WebcamStream />
      <div className="absolute top-4 left-4 z-50">
        <Link href="/dashboard">
          <Button variant="secondary" className="shadow-lg gap-2">
            <LayoutDashboard className="w-4 h-4" />
            Dashboard
          </Button>
        </Link>
      </div>
    </main>
  )
}
