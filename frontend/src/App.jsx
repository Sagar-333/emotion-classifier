import { useState } from 'react';
import { Loader2, Sparkles, AlertCircle, Activity } from 'lucide-react';

export default function App() {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState(null);
  const [status, setStatus] = useState('idle'); 

  const boxConfig = {
    joy: 'bg-emerald-100 text-emerald-800 border-emerald-300 ring-emerald-500',
    anger: 'bg-red-100 text-red-800 border-red-300 ring-red-500',
    sadness: 'bg-blue-100 text-blue-800 border-blue-300 ring-blue-500',
    fear: 'bg-purple-100 text-purple-800 border-purple-300 ring-purple-500',
    disgust: 'bg-orange-100 text-orange-800 border-orange-300 ring-orange-500',
    shame: 'bg-rose-100 text-rose-800 border-rose-300 ring-rose-500',
    guilt: 'bg-yellow-100 text-yellow-800 border-yellow-300 ring-yellow-500',
    default: 'bg-slate-50 text-slate-700 border-slate-200 ring-slate-500'
  };

  const bgConfig = {
    joy: 'bg-emerald-50',
    anger: 'bg-red-50',
    sadness: 'bg-blue-50',
    fear: 'bg-purple-50',
    disgust: 'bg-orange-50',
    shame: 'bg-rose-50',
    guilt: 'bg-yellow-50',
    default: 'bg-slate-50'
  };

  const activeBoxTheme = boxConfig[emotion] || boxConfig.default;
  const activeBgTheme = bgConfig[emotion] || bgConfig.default;

  const analyzeText = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;
    
    setStatus('loading');
    setEmotion(null);

    try {
      const response = await fetch('https://emotion-classifier-zszt.onrender.com/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) throw new Error('API request failed');

      const data = await response.json();
      
  
      const cleanEmotion = String(data.predicted_emotion).toLowerCase().trim();
      
      setEmotion(cleanEmotion);
      setStatus('success');
    } catch (error) {
      console.error("Inference Error:", error);
      setStatus('error');
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center p-4 selection:bg-indigo-100 selection:text-indigo-900 transition-colors duration-700 ${activeBgTheme}`}>
      
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-xl shadow-slate-200/50 border border-slate-100 overflow-hidden">
        
        <div className="border-b border-slate-100 p-6 bg-slate-50/50">
          <div className="flex items-center gap-3 mb-1">
            <div className="p-2 bg-indigo-600 rounded-lg shadow-sm">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-semibold text-slate-900 tracking-tight">
              Sentiment Engine
            </h1>
          </div>
          <p className="text-sm text-slate-500 ml-11">
            Reproduces{" "}
            <a href="https://aclanthology.org/P19-1391/" target="_blank" rel="noopener noreferrer" className="underline hover:text-indigo-600">
              Troiano, Padó & Klinger (2019)
            </a>
            , ACL — trained on ISEAR, enISEAR & deISEAR
          </p>
        </div>

        <div className="p-6">
          <form onSubmit={analyzeText} className="space-y-4">
            <div className="relative">
              <textarea 
                rows="4" 
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter a sentence for the model to analyze..."
                className="w-full p-4 bg-slate-50 border border-slate-200 rounded-xl text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all resize-none shadow-inner"
              />
            </div>

            <button 
              type="submit"
              disabled={status === 'loading' || !text.trim()}
              className="w-full flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 text-white py-3 px-4 rounded-xl font-medium transition-all active:scale-[0.98]"
            >
              {status === 'loading' ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Running Inference...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Analyze Emotion
                </>
              )}
            </button>
          </form>

          {status === 'error' && (
            <div className="mt-6 flex items-center gap-3 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100">
              <AlertCircle className="w-5 h-5" />
              <p className="text-sm font-medium">Failed to connect to the prediction server. Ensure your Express API is running.</p>
            </div>
          )}

          {status === 'success' && emotion && (
            <div className={`mt-6 p-6 rounded-xl border transition-all duration-500 ease-out flex flex-col items-center justify-center gap-2 ${activeBoxTheme}`}>
              <span className="text-sm font-semibold uppercase tracking-wider opacity-70">
                Predicted Classification
              </span>
              <h2 className="text-3xl font-bold capitalize tracking-tight">
                {emotion}
              </h2>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}