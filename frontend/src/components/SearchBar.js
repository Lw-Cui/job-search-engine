import React from "react";

class SearchBar extends React.Component {
  state = { term: "", way: "tfidf" };

  onFormSubmit = event => {
    event.preventDefault();
    this.props.onFormSubmit(this.state.term, this.state.way);
  };

  render() {
    return (
      <div className="ui segment">
        <form className="ui form" onSubmit={this.onFormSubmit}>
          <div className="ui fluid action input left icon">
            <i className="search icon"></i>
            <input
              type="text"
              placeholder="Search..."
              value={this.state.term}
              onChange={e => this.setState({ term: e.target.value })}
            />
            <div>
              <select
                className="ui selection dropdown"
                onChange={e => this.setState({ way: e.target.value })}
              >
                <option value="bm25">BM25</option>
                <option value="tfidf">TF-IDF</option>
                <option value="bert">BERT</option>
              </select>
            </div>
          </div>
        </form>
      </div>
    );
  }
}

export default SearchBar;
