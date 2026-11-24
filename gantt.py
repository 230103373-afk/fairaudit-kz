import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
wps=[("WP1 Data & legal framework",1,5,"#5b7aa8"),("WP2 Audit & explainability engine",3,11,"#1e8449"),
     ("WP3 Mitigation & benchmarking",8,17,"#d68910"),("WP4 Pilot with KZ lender (TRL 5-6)",14,22,"#8e44ad"),
     ("WP5 Management & dissemination",1,24,"#7f8c8d")]
ms=[(5,"M1"),(11,"M2"),(17,"M3"),(22,"M4"),(24,"M5")]
fig,ax=plt.subplots(figsize=(7.2,2.4))
for i,(n,a,b,c) in enumerate(wps[::-1]): ax.barh(i,b-a+1,left=a-0.5,color=c,height=0.55); ax.text(a-0.3,i,n,va="center",fontsize=7,color="white",fontweight="bold")
for m,l in ms: ax.axvline(m+0.5,color="#c0392b",lw=0.7,ls=":"); ax.text(m+0.5,len(wps)-0.35,l,ha="center",fontsize=7,color="#c0392b")
ax.set_yticks([]); ax.set_xlim(0.5,24.5); ax.set_xticks(range(1,25)); ax.tick_params(labelsize=6); ax.set_xlabel("Month",fontsize=7)
ax.spines[["top","right","left"]].set_visible(False); fig.tight_layout(); fig.savefig("results/fig_gantt.png",dpi=220)
