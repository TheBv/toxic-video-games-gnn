import { create } from "zustand";

interface ProjectState {
  annotator: string | null
  setAnnotator: (annotator: string | null) => void
}

export const useAnnotator = create<ProjectState>((set, get) => ({
  annotator: localStorage.getItem("annotator") || null,
  setAnnotator: (annotator: string | null) => {
    localStorage.setItem("annotator", annotator || "")
    set({ annotator: annotator })
  },
}))
