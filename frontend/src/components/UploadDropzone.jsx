// 업로드 드롭존 — Studio·Privacy가 공유한다. 파일 선택 외의 부가 UI(예시·샘플 버튼)는
// children으로 받아 페이지별 문구만 다르게 쓴다.
export default function UploadDropzone({ title, sub, onPick, children }) {
  return (
    <label className="dropzone">
      <input type="file" accept="image/*" hidden onChange={(e) => onPick(e.target.files[0])} />
      <img className="dropzone-mascot" src="/mascot-select.png" alt="" />
      <p className="dropzone-title">{title}</p>
      <p className="dropzone-sub">{sub}</p>
      {children}
    </label>
  )
}
