/* Read result JSON as inert data. Program source and model output are inserted as text. */
const byId=id=>document.getElementById(id);
function el(tag,content,cls){const node=document.createElement(tag);if(content!==undefined)node.textContent=content;if(cls)node.className=cls;return node;}
function grid(columns,rows){const wrapper=el("div",undefined,"scroll"),t=el("table"),head=t.createTHead().insertRow();columns.forEach(c=>head.append(el("th",c)));const body=t.createTBody();rows.forEach(row=>{const tr=body.insertRow();row.forEach(c=>tr.append(el("td",c??"—")));});wrapper.append(t);return wrapper;}
function chart(rows,checkpoints){const root=byId("chart");root.replaceChildren();if(!rows.length){root.append(el("p","The first scheduled validation evaluation will populate this chart.","empty"));return;}
 const ns="http://www.w3.org/2000/svg";function shape(kind,attr,text){const n=document.createElementNS(ns,kind);Object.entries(attr).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;return n;}
 const w=850,h=235,x=n=>51+n/Math.max(1,rows.at(-1).n)*770,y=v=>197-v*164;
 const svg=shape("svg",{viewBox:"0 0 "+w+" "+h,role:"img","aria-label":"Reviewer accuracy and writer reward over scheduled evaluations"});
 [0,.25,.5,.75,1].forEach(t=>{svg.append(shape("line",{x1:51,y1:y(t),x2:821,y2:y(t),stroke:"#dce6e2"}),shape("text",{x:5,y:y(t)+4,fill:"#46615a","font-size":13},Math.round(100*t)+"%"));});
 checkpoints.forEach(n=>svg.append(shape("line",{x1:x(n),x2:x(n),y1:28,y2:198,stroke:"#899c98","stroke-dasharray":"3 3"})));
 [["review_success","review_total","#086f67"],["writer_success","writer_total","#a94c32"]].forEach(([ok,all,color])=>{let path="",last=null;
 rows.filter(r=>r[all]).forEach(r=>{const yy=y(r[ok]/r[all]),xx=x(r.n);path+=last===null?"M "+xx+" "+yy:" H "+xx+" V "+yy;last=yy;});if(path)svg.append(shape("path",{d:path,stroke:color,fill:"none","stroke-width":3}));});
 [0,Math.ceil(rows.at(-1).n/2),rows.at(-1).n].filter((v,i,a)=>a.indexOf(v)===i).forEach(t=>svg.append(shape("text",{x:x(t),y:217,"text-anchor":"middle",fill:"#46615a","font-size":13},String(t))));root.append(svg);}
let db={runs:[],scores:[],examples:{},contrasts:{}},live={};
function render(){const selected=byId("run").value,stored=db.runs.find(r=>r.name===(selected||live.name));
 const r=(!selected&&live.name)?Object.assign({},stored||{},live):stored;
 const metrics=byId("metrics");metrics.replaceChildren();
 byId("status").textContent=r?[r.name,r.updated?"Updated "+new Date(r.updated).toLocaleString():"Published search snapshot"].join(" · "):"Preregistered · Awaiting the first search evaluation";
 const selections=r?.selections||r?.selection||[],rows=r?.trajectory||[],checkpoints=r?.checkpoints||rows.filter((v,i)=>i>0&&v.epoch!==rows[i-1].epoch).map(v=>v.n-1);
 if(r){const c=r.compute||{},nodes=Array.isArray(r.nodes)?r.nodes.length:r.nodes;const vals=[["Validation",r.n+" / "+r.budget],["Archive nodes",nodes],["Selector edits",selections.filter(s=>s.changed?.includes("selector")).length],["Reviewer changes",r.epoch!==undefined?r.epoch-1:checkpoints.length],["Actual model calls",c.actual_calls],["Ollama service hours",((c.service_seconds||0)/3600).toFixed(2)]];
 vals.forEach(([label,value])=>{const m=el("div",undefined,"metric");m.append(el("strong",String(value??0)),el("span",label));metrics.append(m);});}chart(rows,checkpoints);
 const lineage=byId("lineage");lineage.replaceChildren();if(Array.isArray(stored?.nodes))lineage.append(grid(["Node","Inherited from"],stored.nodes.map(n=>[n.id,n.parent])));
 const ledger=byId("choices");ledger.replaceChildren();if(selections.length){ledger.append(grid(["Validation","Selector owner","Chosen parent","Admitted child","Edited roles"],selections.map(s=>[s.n,s.owner||"Fixed rule",s.parent,s.child,s.changed?.join(", ")])));
 (stored?.selection||[]).filter(s=>s.source).forEach(s=>{const block=el("details");block.append(el("summary","Executed at "+s.n+" · "+s.owner),el("pre",s.source));ledger.append(block);});}else ledger.append(el("p","The archive will list each expansion here."));
 const c=db.contrasts?.primary;byId("effect").textContent=c?"Mean editable minus native: "+c.mean_pp.toFixed(2)+" pp (95% interval "+c.low.toFixed(2)+" to "+c.high.toFixed(2)+"). "+c.interpretation+".":(db.status||"Awaiting verified endpoint data.");
 const results=byId("scores");results.replaceChildren();if(db.scores.length)results.append(grid(["Split","Endpoint","Correct","Total","Accuracy","State"],db.scores.map(e=>[e.split,e.name,e.correct,e.n,(100*e.correct/e.n).toFixed(1)+"%",e.complete?"Complete":"In progress"])));
 const auditRows=[];for(const [split,periods] of Object.entries(db.development||{})){for(const label of ["accept","reject"]){const b=periods.before[label],a=periods.after[label];auditRows.push([split,label,b.n,b.acknowledgments,a.acknowledgments]);}}byId("audit").replaceChildren(grid(["Split","Decision","Papers","Before","After"],auditRows));
 const examples=byId("examples");examples.replaceChildren();for(const [arm,roles] of Object.entries(db.examples||{})){for(const [role,e] of Object.entries(roles)){const b=el("details");b.append(el("summary",arm+" · "+role+" · "+e.decision),el("p",e.id),el("pre",e.text));examples.append(b);}}
 if(!examples.children.length)examples.append(el("p","Examples appear after the first recorded outputs."));}
async function refresh(){try{const read=async file=>{const response=await fetch(file,{cache:"no-store"});if(!response.ok)throw Error(response.status);return response.json();};[db,live]=await Promise.all([read("data.json"),read("live.json")]);const select=byId("run"),previous=select.value;
 select.replaceChildren(new Option("Current run",""),...db.runs.map(r=>new Option(r.name,r.name)));select.value=previous;render();}catch(e){byId("status").textContent="Observation data could not be loaded. Retrying in ten seconds.";}}
byId("run").addEventListener("change",render);refresh();setInterval(refresh,10000);
