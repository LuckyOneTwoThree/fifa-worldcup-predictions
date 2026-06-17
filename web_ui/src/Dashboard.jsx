import React, { useState, useEffect } from 'react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import { ShieldAlert, Zap, TrendingUp, CloudRain, Activity, Info, SunSnow, Search, BarChart3, Crosshair, Scale, Siren, AlertOctagon, HeartPulse, Gauge } from 'lucide-react';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// -------------------------------------------------------------
// Component: Ticker / OSINT Feed
// -------------------------------------------------------------
const OsintFeed = ({ data }) => {
  if (!data) return null;
  const { impacts, match_info } = data;
  const rawNews = impacts._raw_news || { home_news: [], away_news: [] };
  const allNews = [...rawNews.home_news, ...rawNews.away_news].sort(() => Math.random() - 0.5);
  
  return (
    <div className="glass-panel flex flex-col h-full border-slate-700/50">
      <div className="bg-dark-900/80 p-3 border-b border-slate-700/50 flex justify-between items-center">
        <h3 className="text-sm font-bold flex items-center gap-2 text-slate-300">
          <Siren className="text-neon-orange w-4 h-4 animate-pulse" /> OSINT 场外情报流
        </h3>
        <span className="text-[10px] font-mono text-neon-green px-2 py-0.5 rounded-full bg-neon-green/10 border border-neon-green/30">LIVE</span>
      </div>
      
      <div className="flex-1 overflow-hidden relative p-4">
        {/* Fading edges */}
        <div className="absolute top-0 left-0 right-0 h-8 bg-gradient-to-b from-dark-900/80 to-transparent z-10 pointer-events-none"></div>
        <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-dark-900/80 to-transparent z-10 pointer-events-none"></div>
        
        <div className="h-full overflow-y-auto custom-scrollbar pr-2 space-y-4">
          
          <div className="flex items-start gap-3">
            <div className="mt-1"><AlertOctagon className="w-4 h-4 text-neon-red" /></div>
            <div>
              <div className="text-xs font-bold text-neon-red mb-1">伤病监控网络 (Injury Radar)</div>
              <div className="text-[11px] text-slate-400 font-mono">
                主队通报: {impacts.injuries_home} 起 | 客队通报: {impacts.injuries_away} 起<br/>
                休息天数: {impacts.fatigue_home} 天 vs {impacts.fatigue_away} 天
              </div>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <div className="mt-1">
              {impacts.weather && impacts.weather.toLowerCase().includes('clear') ? 
                <SunSnow className="w-4 h-4 text-neon-blue" /> : 
                <CloudRain className="w-4 h-4 text-neon-orange" />}
            </div>
            <div>
              <div className="text-xs font-bold text-neon-blue mb-1">气象对冲中心 (Weather Ops)</div>
              <div className="text-[11px] text-slate-400 font-mono">{impacts.weather}</div>
            </div>
          </div>

          <div className="flex items-start gap-3">
            <div className="mt-1"><Zap className="w-4 h-4 text-neon-purple" /></div>
            <div>
              <div className="text-xs font-bold text-neon-purple mb-1">博弈论推演 (Game Theory)</div>
              <div className="text-[11px] text-slate-400 font-mono">
                动机等级: {impacts.motivation_index} <br/>
                默契球风险: {impacts.biscotto_risk}
              </div>
            </div>
          </div>

          {allNews.map((news, idx) => (
            <div key={idx} className="flex items-start gap-3 border-t border-slate-700/30 pt-3">
              <div className="mt-1 text-[10px] text-slate-500 font-mono">{new Date().toISOString().substring(11,16)}</div>
              <div>
                <div className="text-[11px] font-bold text-slate-300 mb-1 leading-tight">{news.title}</div>
                <div className="text-[10px] text-slate-500 leading-tight line-clamp-3">{news.body}</div>
              </div>
            </div>
          ))}

        </div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// Component: Poisson Heatmap
// -------------------------------------------------------------
const PoissonHeatmap = ({ data }) => {
  if (!data) return null;
  const scores = data.match_info.score_probs;
  
  const matrix = Array(4).fill(0).map(() => Array(4).fill(null));
  let maxProb = 0;
  
  scores.forEach(s => {
    const [h, a] = s.score.split('-').map(Number);
    if (h < 4 && a < 4) {
      matrix[h][a] = s.prob;
      if (s.prob > maxProb) maxProb = s.prob;
    }
  });

  const getHeatColor = (prob) => {
    if (!prob) return 'bg-dark-950 border-slate-800 text-slate-700';
    const ratio = prob / maxProb;
    if (ratio > 0.8) return 'bg-neon-red/40 border-neon-red text-white shadow-[0_0_15px_rgba(248,113,113,0.3)]';
    if (ratio > 0.5) return 'bg-neon-orange/30 border-neon-orange/80 text-slate-200';
    if (ratio > 0.2) return 'bg-neon-blue/20 border-neon-blue/50 text-slate-300';
    return 'bg-dark-800/80 border-slate-700 text-slate-400';
  };

  return (
    <div className="glass-panel h-full flex flex-col border-slate-700/50">
      <div className="bg-dark-900/80 p-3 border-b border-slate-700/50">
        <h3 className="text-sm font-bold flex items-center gap-2 text-slate-300">
          <TrendingUp className="text-neon-red w-4 h-4" /> 泊松分布波胆矩阵
        </h3>
      </div>
      <div className="p-4 flex-1 flex flex-col justify-center">
        <div className="grid grid-cols-5 gap-1 text-center max-w-[300px] mx-auto">
          <div className="col-span-1"></div>
          {[0,1,2,3].map(a => <div key={`col-${a}`} className="text-[10px] font-mono text-slate-500 mb-1">客 {a}</div>)}
          
          {[0,1,2,3].map(h => (
            <React.Fragment key={`row-${h}`}>
              <div className="flex items-center justify-center text-[10px] font-mono text-slate-500 pr-1">主 {h}</div>
              {matrix[h].map((prob, a) => (
                <div 
                  key={`${h}-${a}`} 
                  className={cn(
                    "aspect-square rounded border flex flex-col items-center justify-center transition-all duration-300",
                    getHeatColor(prob)
                  )}
                >
                  <div className="font-bold text-sm tracking-tighter">{h}:{a}</div>
                  <div className="text-[9px] opacity-70 font-mono">{prob ? (prob * 100).toFixed(1) + '%' : '-'}</div>
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// Component: Radar Chart
// -------------------------------------------------------------
const RadarChartPanel = ({ match, data }) => {
  if (!data) return null;
  const { match_info, impacts } = data;
  const [t1, t2] = match.split(' vs ');

  const normElo = (val) => Math.max(0, Math.min(100, (val - 1400) / 8));
  const maxSv = Math.max(match_info.sv_home, match_info.sv_away) || 1;
  const normSv = (val) => (val / maxSv) * 100;
  const normPpda = (val) => Math.max(0, 100 - (val * 3));
  const normHealth = (inj) => Math.max(0, 100 - (inj * 30));
  const normFatigue = (rest) => Math.min(100, rest * 12.5);
  const maxXg = Math.max(match_info.xg_home, match_info.xg_away) || 1;
  const normXg = (val) => (val / maxXg) * 100;

  const radarData = [
    { subject: '实力底蕴(Elo)', A: normElo(match_info.elo_home), B: normElo(match_info.elo_away), fullMark: 100 },
    { subject: '阵容身价(Value)', A: normSv(match_info.sv_home), B: normSv(match_info.sv_away), fullMark: 100 },
    { subject: '前场压迫(PPDA)', A: normPpda(match_info.ppda_home), B: normPpda(match_info.ppda_away), fullMark: 100 },
    { subject: '防空能力(Aerial)', A: match_info.aerial_home || 50, B: match_info.aerial_away || 50, fullMark: 100 },
    { subject: '期望进球(xG)', A: normXg(match_info.xg_home), B: normXg(match_info.xg_away), fullMark: 100 },
    { subject: '健康状态(Health)', A: normHealth(impacts.injuries_home), B: normHealth(impacts.injuries_away), fullMark: 100 },
    { subject: '体能储备(Stamina)', A: normFatigue(impacts.fatigue_home), B: normFatigue(impacts.fatigue_away), fullMark: 100 },
  ];

  return (
    <div className="glass-panel flex flex-col h-full border-slate-700/50">
      <div className="bg-dark-900/80 p-3 border-b border-slate-700/50">
        <h3 className="text-sm font-bold flex items-center gap-2 text-slate-300">
          <Crosshair className="text-neon-blue w-4 h-4" /> V10.5 九维战力模型
        </h3>
      </div>
      <div className="flex-1 min-h-[250px] relative">
        {/* Glow behind radar */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 bg-neon-blue/20 rounded-full blur-3xl"></div>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="65%" data={radarData}>
            <PolarGrid stroke="rgba(51, 65, 85, 0.5)" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'monospace' }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
            <RechartsTooltip contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: '#334155', color: '#fff', fontSize: '12px' }} />
            <Radar name={t1} dataKey="A" stroke="#38bdf8" strokeWidth={2} fill="#38bdf8" fillOpacity={0.3} />
            <Radar name={t2} dataKey="B" stroke="#f87171" strokeWidth={2} fill="#f87171" fillOpacity={0.3} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      <div className="p-3 border-t border-slate-700/50 flex justify-center gap-6 text-[11px] font-mono">
        <div className="flex items-center gap-2"><div className="w-2 h-2 bg-neon-blue rounded-full shadow-[0_0_8px_#38bdf8]"></div>{t1}</div>
        <div className="flex items-center gap-2"><div className="w-2 h-2 bg-neon-red rounded-full shadow-[0_0_8px_#f87171]"></div>{t2}</div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// Component: Core Stats Bar
// -------------------------------------------------------------
const StatBar = ({ label, val1, val2, suffix = '', precision = 1, reverseColors = false }) => {
  const total = Number(val1) + Number(val2) || 1;
  const p1 = (Number(val1) / total) * 100;
  const p2 = (Number(val2) / total) * 100;
  const c1 = reverseColors ? 'bg-neon-red' : 'bg-neon-blue';
  const c2 = reverseColors ? 'bg-neon-blue' : 'bg-neon-red';

  return (
    <div className="mb-3">
      <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
        <span>{Number(val1).toFixed(precision)}{suffix}</span>
        <span className="text-slate-500">{label}</span>
        <span>{Number(val2).toFixed(precision)}{suffix}</span>
      </div>
      <div className="h-1.5 w-full bg-dark-950 rounded-full overflow-hidden flex">
        <div className={`${c1} h-full`} style={{ width: `${p1}%` }}></div>
        <div className={`${c2} h-full`} style={{ width: `${p2}%` }}></div>
      </div>
    </div>
  );
};

// -------------------------------------------------------------
// Main Dashboard
// -------------------------------------------------------------
export default function Dashboard() {
  const [intel, setIntel] = useState(null);
  const [selectedMatch, setSelectedMatch] = useState('');

  useEffect(() => {
    fetch('/latest_intel.json')
      .then(res => res.json())
      .then(data => {
        setIntel(data);
        setSelectedMatch(Object.keys(data)[0]);
      })
      .catch(err => console.error("Failed to load intel", err));
  }, []);

  if (!intel || !selectedMatch) return (
    <div className="min-h-screen bg-dark-950 flex flex-col items-center justify-center font-mono">
      <div className="relative w-24 h-24 mb-8">
        <div className="absolute inset-0 border-t-2 border-neon-blue rounded-full animate-spin"></div>
        <div className="absolute inset-2 border-r-2 border-neon-red rounded-full animate-spin animation-delay-150"></div>
        <div className="absolute inset-4 border-b-2 border-neon-green rounded-full animate-spin animation-delay-300"></div>
        <Activity className="absolute inset-0 m-auto w-6 h-6 text-white animate-pulse" />
      </div>
      <div className="text-neon-blue text-sm tracking-[0.3em]">正在初始化 V10.5 量化引擎...</div>
    </div>
  );

  const matchData = intel[selectedMatch];
  const { match_info } = matchData;
  const [t1, t2] = selectedMatch.split(' vs ');

  return (
    <div className="min-h-screen bg-dark-950 flex flex-col lg:flex-row font-mono text-slate-200">
      
      {/* ---------------- Sidebar ---------------- */}
      <aside className="w-full lg:w-64 lg:h-screen lg:sticky lg:top-0 bg-dark-900/50 border-b lg:border-b-0 lg:border-r border-slate-800 flex flex-col z-20 shrink-0">
        <div className="p-5 border-b border-slate-800">
          <h1 className="text-xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-neon-blue to-white">
            ANTIGRAVITY
          </h1>
          <div className="text-[9px] text-neon-green tracking-widest mt-1 flex items-center gap-1">
            <span className="w-1.5 h-1.5 bg-neon-green rounded-full animate-pulse"></span>
            V10.5 HEDGE DESK
          </div>
        </div>
        
        <div className="p-3 text-[10px] text-slate-500 font-bold uppercase tracking-wider">对阵列表 (Matches)</div>
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          {Object.keys(intel).map(m => {
            const isActive = selectedMatch === m;
            const teams = m.split(' vs ');
            return (
              <button
                key={m}
                onClick={() => setSelectedMatch(m)}
                className={cn(
                  "w-full text-left px-4 py-3 border-l-2 transition-all group",
                  isActive 
                    ? "border-neon-blue bg-neon-blue/5" 
                    : "border-transparent hover:bg-dark-800/50"
                )}
              >
                <div className="flex justify-between items-center mb-1">
                  <div className={cn("text-xs font-bold", isActive ? "text-white" : "text-slate-400")}>{teams[0]}</div>
                  <div className="text-[9px] px-1 py-0.5 bg-dark-950 text-slate-500 rounded">vs</div>
                  <div className={cn("text-xs font-bold", isActive ? "text-white" : "text-slate-400")}>{teams[1]}</div>
                </div>
                {/* Micro preview indicator */}
                <div className="h-0.5 w-full bg-dark-950 rounded overflow-hidden flex">
                  <div className="bg-neon-blue h-full" style={{width: `${intel[m].match_info.p_home*100}%`}}></div>
                  <div className="bg-slate-700 h-full" style={{width: `${intel[m].match_info.p_draw*100}%`}}></div>
                  <div className="bg-neon-red h-full" style={{width: `${intel[m].match_info.p_away*100}%`}}></div>
                </div>
              </button>
            );
          })}
        </div>
      </aside>

      {/* ---------------- Main Content ---------------- */}
      <main className="flex-1 p-4 md:p-6 lg:h-screen lg:overflow-y-auto custom-scrollbar flex flex-col gap-6">
        
        {/* Top Header Row */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2 py-0.5 bg-neon-blue/20 text-neon-blue border border-neon-blue/30 text-[10px] rounded">
                {match_info.tier_home}
              </span>
              <span className="text-slate-500 text-xs">vs</span>
              <span className="px-2 py-0.5 bg-neon-red/20 text-neon-red border border-neon-red/30 text-[10px] rounded">
                {match_info.tier_away}
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight">
              <span className="text-neon-blue">{t1}</span> <span className="text-slate-600 font-normal">v</span> <span className="text-neon-red">{t2}</span>
            </h2>
          </div>
          
          {/* Referee & Odds Info */}
          <div className="flex gap-4">
            <div className="glass-panel px-4 py-2 border-slate-700/50 flex flex-col items-end">
              <div className="text-[10px] text-slate-500 mb-1 flex items-center gap-1"><Scale className="w-3 h-3"/> 执法裁判 (严厉度)</div>
              <div className="text-sm font-bold text-slate-300">{match_info.referee || '未知'}</div>
              <div className="w-full h-1 bg-dark-950 mt-1 rounded overflow-hidden">
                <div className={cn("h-full", match_info.strictness > 0.6 ? "bg-neon-red" : "bg-neon-green")} style={{width: `${(match_info.strictness||0.5)*100}%`}}></div>
              </div>
            </div>
          </div>
        </header>

        {/* Data Dense Row 1 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="glass-panel p-4 border-slate-700/50">
            <div className="text-[10px] text-slate-500 mb-3 flex items-center gap-1"><BarChart3 className="w-3 h-3"/> 核心基本面 (Fundamentals)</div>
            <StatBar label="绝对实力 (Elo)" val1={match_info.elo_home} val2={match_info.elo_away} />
            <StatBar label="全队身价 (亿欧)" val1={match_info.sv_home} val2={match_info.sv_away} />
            <StatBar label="理论进球 (xG Base)" val1={match_info.xg_home} val2={match_info.xg_away} precision={2} />
          </div>
          <div className="glass-panel p-4 border-slate-700/50">
            <div className="text-[10px] text-slate-500 mb-3 flex items-center gap-1"><Gauge className="w-3 h-3"/> 战术压制力 (Tactics)</div>
            <StatBar label="预计控球率 (%)" val1={match_info.possession_home_norm} val2={match_info.possession_away_norm} />
            <StatBar label="前场逼抢强度 (PPDA)" val1={match_info.ppda_home} val2={match_info.ppda_away} reverseColors={true} />
            <StatBar label="高空球胜率" val1={match_info.aerial_home || 50} val2={match_info.aerial_away || 50} />
          </div>
          
          <div className="glass-panel p-4 border-slate-700/50 col-span-1 lg:col-span-2 flex flex-col justify-center relative overflow-hidden">
            <div className="absolute -right-10 -bottom-10 opacity-5"><Zap className="w-48 h-48"/></div>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-[10px] text-slate-500 mb-1">主胜概率</div>
                <div className="text-2xl font-black text-neon-blue">{(match_info.p_home * 100).toFixed(1)}<span className="text-sm">%</span></div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-slate-500 mb-1">平局概率</div>
                <div className="text-2xl font-black text-slate-400">{(match_info.p_draw * 100).toFixed(1)}<span className="text-sm">%</span></div>
              </div>
              <div className="text-center">
                <div className="text-[10px] text-slate-500 mb-1">客胜概率</div>
                <div className="text-2xl font-black text-neon-red">{(match_info.p_away * 100).toFixed(1)}<span className="text-sm">%</span></div>
              </div>
            </div>
          </div>
        </div>

        {/* Multi-Panel Row 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-1 h-[400px] lg:h-[380px]"><RadarChartPanel match={selectedMatch} data={matchData} /></div>
          <div className="lg:col-span-1 h-[350px] lg:h-[380px]"><PoissonHeatmap data={matchData} /></div>
          <div className="lg:col-span-1 h-[350px] lg:h-[380px]"><OsintFeed data={matchData} /></div>
        </div>

        {/* Execution Terminal */}
        <div className="glass-panel p-4 border-slate-700/50 mt-2 shrink-0">
          <div className="flex items-center gap-2 mb-4 pb-2 border-b border-slate-700/50">
            <Activity className="w-4 h-4 text-neon-green" />
            <h3 className="text-sm font-bold text-white">交易执行终端 (Execution Terminal)</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] text-slate-500">亚洲盘口 (ASIAN HANDICAP)</span>
              <div className="text-lg font-bold text-neon-orange font-sans">{match_info.ah_str || "暂无盘口"}</div>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] text-slate-500">大小球 (OVER / UNDER)</span>
              <div className="text-lg font-bold text-neon-purple font-sans">{match_info.ou_str || "暂无盘口"}</div>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] text-slate-500">引擎置信度 (CONFIDENCE)</span>
              <div className="flex items-center gap-2">
                <div className={cn("w-3 h-3 rounded-full animate-pulse", match_info.confidence_index === 'HIGH' ? 'bg-neon-green' : 'bg-yellow-500')}></div>
                <div className="text-lg font-bold text-white">{match_info.confidence_index === 'HIGH' ? '极高 (HIGH)' : '中等 (MEDIUM)'}</div>
              </div>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
