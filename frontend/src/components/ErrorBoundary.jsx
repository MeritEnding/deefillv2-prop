import { Component } from 'react'
import { Icon } from './icons.jsx'

// 에디터에서 예외가 나도 흰 화면 대신 복구 UI를 보여준다.
export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('편집 중 오류:', error, info)
  }

  reset = () => {
    this.setState({ error: null })
    this.props.onReset?.()
  }

  render() {
    if (this.state.error) {
      return (
        <div className="crash">
          <div className="crash-icon"><Icon.warn width="42" height="42" /></div>
          <h3>문제가 발생했어요</h3>
          <p>편집 중 일시적인 오류가 났습니다. 다시 시도하면 됩니다.</p>
          <button className="btn btn-primary" onClick={this.reset}>다시 시작</button>
        </div>
      )
    }
    return this.props.children
  }
}
