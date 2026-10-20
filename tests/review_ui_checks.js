// Execute generated UI logic with a deterministic DOM double; no browser/network.
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const html=fs.readFileSync(process.argv[2],'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
function run(initial='null',failWrite=false){
  const elements={}, saved={}, callbacks=[];
  const ctx=new Proxy({}, {get:(o,k)=>o[k]||(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
  const element=id=>elements[id] ||= {value:'',textContent:'',disabled:false,currentTime:0,children:[],getContext:()=>ctx,append(x){this.children.push(x)},pause(){},load(){},removeAttribute(){},click(){}};
  const context={document:{getElementById:element,createElement:()=>element('created'+Object.keys(elements).length)},localStorage:{getItem:()=>initial,setItem(k,v){if(failWrite)throw Error('disabled');saved[k]=v}},performance:{now:()=>10},URL:{createObjectURL:()=>'',revokeObjectURL(){}},Blob,Date,setTimeout:f=>f()};
  element('video').requestVideoFrameCallback=f=>callbacks.push(f);
  vm.createContext(context);vm.runInContext(script,context);
  return {elements,saved,context,callbacks};
}
let a=run();
a.elements.frame.value='1';a.elements.frame.oninput();assert.match(a.elements.prediction.textContent,/Frame 1 · model A/);
a.elements.notes.value='Keep this evidence';a.elements.notes.oninput();
assert.ok(Object.keys(a.saved)[0].includes('test-candidate'));
a.elements.video.onloadeddata();const stale=a.callbacks[0];
a.elements.next.onclick();stale(1,{mediaTime:.6});assert.match(a.elements.prediction.textContent,/Frame 0/);assert.equal(a.elements.notes.value,'');
a.elements.prev.onclick();assert.equal(a.elements.notes.value,'Keep this evidence');
assert.match(a.elements.prediction.textContent,/Frame 0 · model H/);
assert.equal(vm.runInContext('frameFromTime(.039,D.queries[0])',a.context),0);
assert.equal(vm.runInContext('frameFromTime(.04,D.queries[0])',a.context),1);
let b=run('{}',true);b.elements.notes.value='Retain in memory';b.elements.notes.oninput();assert.match(b.elements.saved.textContent,/remain in memory/);
b.elements.next.onclick();b.elements.prev.onclick();assert.equal(b.elements.notes.value,'Retain in memory');
for(const invalid of ['null','[]','{broken','{"10":null}'])assert.doesNotThrow(()=>run(invalid));
console.log('Viewer checks passed: frame timing, exact labels, autosave/navigation, storage isolation and recovery.');
