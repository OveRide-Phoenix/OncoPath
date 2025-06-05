"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Brain, Upload, Activity, RotateCcw, User } from "lucide-react"

interface NavigationProps {
  onReset: () => void
  currentState: "upload" | "processing" | "results"
}

export function Navigation({ onReset, currentState }: NavigationProps) {
  const getStateInfo = () => {
    switch (currentState) {
      case "upload":
        return { icon: Upload, text: "Upload", color: "bg-blue-500/20 text-blue-300" }
      case "processing":
        return { icon: Activity, text: "Processing", color: "bg-yellow-500/20 text-yellow-300" }
      case "results":
        return { icon: Activity, text: "Results", color: "bg-green-500/20 text-green-300" }
    }
  }

  const stateInfo = getStateInfo()
  const StateIcon = stateInfo.icon

  return (
    <nav className="bg-slate-900/95 backdrop-blur-sm border-b border-slate-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <span className="text-xl font-bold text-white">OncoPath</span>
              <Badge variant="secondary" className="bg-blue-500/20 text-blue-300 border-blue-500/30">
                v2.1
              </Badge>
            </div>

            <div className="hidden md:flex items-center space-x-2">
              <div className="w-px h-6 bg-slate-600"></div>
              <Badge variant="secondary" className={stateInfo.color}>
                <StateIcon className="h-3 w-3 mr-1" />
                {stateInfo.text}
              </Badge>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {currentState !== "upload" && (
              <Button
                variant="outline"
                size="sm"
                onClick={onReset}
                className="border-slate-600 text-slate-300 hover:text-white hover:bg-slate-800"
              >
                <RotateCcw className="h-4 w-4 mr-2" />
                New Analysis
              </Button>
            )}

            <Button variant="ghost" size="sm" className="text-slate-300 hover:text-white">
              <User className="h-4 w-4 mr-2" />
              Dr. Smith
            </Button>
          </div>
        </div>
      </div>
    </nav>
  )
}
