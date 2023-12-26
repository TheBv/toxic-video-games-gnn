import { useCallback, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import NavBar from './components/NavBar';
import HelpPage from './pages/HelpPage';
import AnnotatorPage from './pages/AnnotatorPage';
import ProjectsPage from './pages/ProjectsPage';
import { AnnotateRoute } from './pages/toxic_games/ToxicGameAnnotation';
import LandingPage from './pages/LandingPage';


export default function App() {

  // List of annotation tasks/projects
  const [currentProject, setCurrentProject] = useState(null)

  const ListComponent = useCallback(() => {
    if (!currentProject) {
      return null;
    }

    switch (currentProject["key"]) {
      case "biofid_srl":
        return <p>Biofid SRL List</p>;
      default:
        return <p>This project does not support listing all annotations.</p>;
    }
  }, [currentProject])


  return (
    <>
      <div className="App">
        <NavBar />
      </div>
      <Routes>
        <Route path="/" element={LandingPage()} />.
        <Route path="/projects" element={ProjectsPage()}/>
        <Route path="/annotation" element={AnnotateRoute()}/>
        <Route path="/user" element={AnnotatorPage()} />
        <Route path="/list" element={ListComponent()} />
        <Route path="/help" element={HelpPage()} />
      </Routes>
    </>
  )
}
