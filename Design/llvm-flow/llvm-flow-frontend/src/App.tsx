import {
  BrowserRouter as Router,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom'
import Main from '@/components/pages/main/main'
import Upload from '@/components/pages/upload/upload'
import Profile from '@/components/pages/profile/profile'
import LLVMcfg from '@/components/pages/llvmcfg/llvmcfg'
import Tutorial from '@/components/pages/tutorial/Tutorial'
import NavBar from './components/modules/navbar/NavBar'
import Footer from './components/modules/footer/Footer'
import { useAppDispatch } from '@/redux/hook'
import { setAuthData } from '@/redux/features/auth/authSlice'
import { useEffect } from 'react'

function App() {
  const dispatch = useAppDispatch()

  useEffect(() => {
    const data = localStorage.getItem('nickname')
    if (data) {
      dispatch(setAuthData(JSON.parse(data)))
    }
  })

  return <Router><AppContent /></Router>
}

function AppContent() {
  const location = useLocation()
  const isWorkbench = location.pathname === '/llvmcfg'

  return (
    <>
      {!isWorkbench && <NavBar />}
      <Routes>
        <Route path="/" element={<Main />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/tutorial" element={<Tutorial />} />
        <Route path="/board" element={<Profile />} />
        <Route path="/llvmcfg" element={<LLVMcfg />} />
      </Routes>
      {!isWorkbench && <Footer />}
    </>
  )
}

export default App
