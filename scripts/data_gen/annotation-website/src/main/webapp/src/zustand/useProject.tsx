import { create } from "zustand"
import { BASE_URL } from "../lib/constants"

interface ProjectState {
  currentProject: Project | null
  projectList: Project[]
  setCurrentProject: (project: Project) => void
  fetchProjects: (status?: ProjectStatus) => Promise<Project[]>
}

export type ProjectStatus = null | "PREVIEW" | "OPEN" | "CLOSED"

export interface Project {
  key: string
  name: string
  description: string
  languages: string[]
  authors: string[]
  status: ProjectStatus
  start_at: string
  end_at: string
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  currentProject: null,
  projectList: [],
  setCurrentProject: (project: Project) => {
    set({ currentProject: project })
  },
  fetchProjects: async (status) => {
    const params = new URLSearchParams()
    
    if (status)
      params.set("status", status)

    fetch(BASE_URL + "/api/v1/projects?" + params, {
      method: "GET",
    }).then(async response => {

      if (response.status === 200) {
        const data = await response.json()
        const projects = data["projects"]
        set({ projectList: projects })
        return projects
      }
      return []
    }).catch(error => console.error(error))
    return []
  }
}))