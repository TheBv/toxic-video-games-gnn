import { useEffect, useState } from "react";
import { useAnnotator } from "../zustand/useAnnotator";
import { BASE_URL } from '../lib/constants';

interface AnnotatedCount {
  total: number
  self: number
}

export default function useAnnotatedCount() {
  const { annotator } = useAnnotator()
  const [count, setCount] = useState<AnnotatedCount>({ total: 0, self: 0 })

  const fetchAnnotatedCount = async () => {
    const payload = {
      annotator: annotator
    }
    const result = await fetch(BASE_URL + "/api/v1/toxic_games/count", {
      method: 'POST',
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload)
    })
    const json = await result.json()
    setCount({
      total: json.total_count,
      self: json.self
    })

  }

  useEffect(() => {
    fetchAnnotatedCount()
  }, [annotator])

  return { count, setCount }
}